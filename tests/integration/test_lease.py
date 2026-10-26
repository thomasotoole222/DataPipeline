"""Integration tests for ``pipeline.worker.lease`` against a database with control loaded."""

import os
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any
from uuid import UUID

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from psycopg_pool import AsyncConnectionPool

from pipeline.db.control import ClaimedTask, LeaseLoss, TaskReport, claim
from pipeline.db.pool import Pool
from pipeline.errors import LeaseConflict, Outcome
from pipeline.telemetry import Instruments
from pipeline.worker.lease import Lease

DSN = os.environ.get("PIPELINE_TEST_DSN", "postgresql://postgres@localhost:5432/pipeline_test")
SOURCE = UUID("00000000-0000-0000-0000-000000000001")
WORKER = "test-worker"


@pytest.fixture
async def pool() -> AsyncIterator[Pool]:
    """Yield an open one-connection pool on empty control tables."""
    p: Pool = AsyncConnectionPool(DSN, min_size=1, max_size=1, kwargs={"autocommit": True}, open=False)
    await p.open(wait=True)
    async with p.connection() as c:
        await c.execute("TRUNCATE control.lease_loss, control.task RESTART IDENTITY")
    try:
        yield p
    finally:
        await p.close()


@pytest.fixture
def reader() -> InMemoryMetricReader:
    """Return an in-memory metric reader."""
    return InMemoryMetricReader()


@pytest.fixture
def instruments(reader: InMemoryMetricReader) -> Instruments:
    """Build the pipeline's instruments on a meter provider read by ``reader``."""
    meter = MeterProvider(metric_readers=[reader]).get_meter("test")
    return Instruments(
        task_duration=meter.create_histogram("pipeline.task.duration", unit="s"),
        rows=meter.create_counter("pipeline.rows"),
        connector_requests=meter.create_counter("pipeline.connector.requests"),
        log_dropped=meter.create_counter("pipeline.log.dropped"),
        slots_busy=meter.create_up_down_counter("pipeline.worker.slots_busy"),
        queue_wait=meter.create_histogram("pipeline.task.queue_wait", unit="s"),
        heartbeat_lag=meter.create_histogram("pipeline.lease.heartbeat_lag", unit="s"),
        lease_losses=meter.create_counter("pipeline.lease.losses"),
        lease_unwind=meter.create_histogram("pipeline.lease.unwind", unit="s"),
        unknown_jobs=meter.create_counter("pipeline.task.unknown_job"),
        worker_paused=meter.create_counter("pipeline.worker.paused"),
    )


def points(reader: InMemoryMetricReader, name: str) -> list[Any]:
    """Return the data points recorded for the metric ``name``."""
    data = reader.get_metrics_data()
    if data is None:
        return []
    return [
        point
        for rm in data.resource_metrics
        for sm in rm.scope_metrics
        for metric in sm.metrics
        if metric.name == name
        for point in metric.data.data_points
    ]


def success(duration: timedelta = timedelta(seconds=1)) -> TaskReport:
    """Build a succeeded report with no rows."""
    return TaskReport(
        outcome=Outcome.SUCCEEDED,
        error_class=None,
        error_message=None,
        retry_after=None,
        rows_fetched=0,
        rows_written=0,
        rows_rejected=0,
        duration=duration,
    )


async def claimed(pool: Pool) -> ClaimedTask:
    """Insert a pending orders task and claim it."""
    async with pool.connection() as c:
        await c.execute(
            "INSERT INTO control.task (run_id, connector, job, source_id, partition_key) "
            "VALUES (7, 'tiktok_shop', 'orders', %s, '2026-10-19')",
            (SOURCE,),
        )
        task = await claim(c, WORKER)
    assert task is not None
    return task


async def steal(pool: Pool, task: ClaimedTask) -> None:
    """Move the task to another lease, as a re-claim by another worker would."""
    async with pool.connection() as c:
        await c.execute("UPDATE control.task SET lease_token = gen_random_uuid() WHERE id = %s", (task.id,))


async def test_complete_with_a_stale_token_records_a_commit_loss(
    pool: Pool, instruments: Instruments, reader: InMemoryMetricReader
) -> None:
    """Check that complete on a stolen lease raises, records a commit loss and leaves completed unset."""
    task = await claimed(pool)
    held = Lease(pool, task, worker_id=WORKER, instruments=instruments)
    await steal(pool, task)
    async with pool.connection() as c:
        with pytest.raises(LeaseConflict):
            await held.complete(c, success())
    assert held.loss is LeaseLoss.COMMIT
    assert held.lost.is_set()
    assert not held.completed
    assert points(reader, "pipeline.task.duration") == []


async def test_complete_records_the_task_duration(
    pool: Pool, instruments: Instruments, reader: InMemoryMetricReader
) -> None:
    """Check that complete records the reported duration, with the job and outcome as attributes."""
    task = await claimed(pool)
    held = Lease(pool, task, worker_id=WORKER, instruments=instruments)
    async with pool.connection() as c:
        await held.complete(c, success(duration=timedelta(seconds=2.5)))
    (point,) = points(reader, "pipeline.task.duration")
    assert (point.sum, point.count) == (2.5, 1)
    assert dict(point.attributes) == {"job": "orders", "outcome": "succeeded"}
