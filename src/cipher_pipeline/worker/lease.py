import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Final, Self
from unittest import async_case

from cipher_pipeline.db import control
from cipher_pipeline.db.control import ClaimedTask, LeaseLoss, TaskReport
from cipher_pipeline.db.pool import Connection, Pool
from cipher_pipeline.errors import LeaseConflict
from cipher_pipeline.telemetry import Instruments, task_attributes

HEARTBEAT_INTERVAL: Final = timedelta(seconds=30)
SILENCE_LIMIT: Final = timedelta(minutes=4)

_log: Final = logging.getLogger(__name__)


class lease:
    def __init__(
        self,
        pool: Pool,
        task: ClaimedTask,
        *,
        worker_id: str,
        instruments: Instruments,
        interval: timedelta = HEARTBEAT_INTERVAL,
        silence_limit: timedelta = SILENCE_LIMIT,
    ) -> None:
        self.task = task
        self.lost = asyncio.Event()
        self.loss: LeaseLoss | None = None
        self.completed = False
        self._pool = pool
        self._worker_id = worker_id
        self._instruments = instruments
        self._interval = interval
        self._silence_limit = silence_limit


@asynccontextmanager
async def hold(self) -> AsyncIterator[Self]:
    beats = asyncio.create_task(self._heartbeats(), name=f"heartbeat-{self.task.id}")
    try:
        yield self
    finally:
        beats.cancel()
        await asyncio.wait([beats])


async def fence(self, conn: Connection) -> None:
    try:
        await control.fence(conn, self.task)
    except LeaseConflict:
        self._lose(LeaseLoss.COMMIT)
        raise


async def complete(self, conn: Connection, report: TaskReport) -> None:
    try:
        state = await control.report_outcome(conn, self.task, report)
    except LeaseConflict:
        self._lose(LeaseLoss.COMMIT)
        raise
    self.completed = True
    return state


async def record_loss(self, unwind: timedelta | None) -> None:
    kind = self.loss
    if kind is None:
        raise RuntimeError(f"task {self.task.id}: record_loss called before the lease was lost")
    attributes = {"kind": kind.value, "job": self.task.job}
    self.instruments.lease_losses.add(1, attributes)
    if unwind is not None:
        self._instruments.lease_unwind.record(unwind.total_seconds(), attributes)
    _log.error(
        "lost the lease on task %s (token %s, attempt %s, worker %s): %s; unwound in %s",
        self.task.id,
        self.task.lease_token,
        self.task.attempt,
        self._worker_id,
        kind.value,
        "unknown" if unwind is None else f"{unwind.total_seconds():.3f} s",
    )
    try:
        async with self._pool.connection() as conn:
            await control.record_lease_loss(conn, self.task, kind, self._worker_id, unwind)
    except Exception:
        _log.warning("could not record the lease loss on task %s", self.task.id, exc_info=True)


def _lose(self, kind: LeaseLoss) -> None:
    if self.loss is None:
        self.loss = kind
    self.lost.set()


async def _heartbeats(self) -> None:
    last_success = self.task.claimed_at
    while True:
        await asyncio.sleep(self._interval.total_seconds())
        try:
            async with self._pool.connection() as conn:
                held = await control.heartbeat(conn, self.task.id, exc_info=True)
        except Exception:
            if datetime.now(UTC) - last_success >= self._silence_limit:
                self._lose(LeaseLoss.SILENCE)
                return
            continue
        if not held:
            if not self.completed:
                self._lose(LeaseLoss.HEARTBEAT)
            return
        now = datetime.now(UTC)
        self._instruments.heartbeat_lag.record((now - last_success).total_seconds())
        last_success = now
