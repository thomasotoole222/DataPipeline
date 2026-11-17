"""Integration tests for the raw archive: ``pipeline.archive.months`` and ``pipeline.jobs.archive_raw``."""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import polars as pl
import psycopg
import pytest
from psycopg_pool import AsyncConnectionPool

from pipeline.archive.months import (
    ArchiveCheckFailed,
    MonthKey,
    archive_month,
    check_manifest,
    object_key,
    plan_months,
)
from pipeline.archive.store import DirectoryStore, StoredObject
from pipeline.db.control import ClaimedTask, TaskReport
from pipeline.db.pool import Connection, Pool
from pipeline.db.raw import RawPage, save_page
from pipeline.errors import Outcome
from pipeline.jobs.archive_raw import ArchiveNotConfigured, ArchiveRawJob

DSN = os.environ.get("TEST_PIPELINE_DSN", "postgresql://postgres@localhost:5432/pipeline_test")

pytestmark = pytest.mark.integration

SHOP_A = UUID("00000000-0000-0000-0000-00000000000a")
SHOP_B = UUID("00000000-0000-0000-0000-00000000000b")
KEYS = frozenset({"data"})  # the hashed keys, as TikTok Shop's
MAY = date(2026, 5, 1)
TODAY = date(2026, 6, 10)  # May is settled for a job re-fetched up to 7 days back (the 10th), not for 30 days


@pytest.fixture
async def pool() -> AsyncIterator[Pool]:
    """Yield an open pool on empty raw, manifest and task tables."""
    p: Pool = AsyncConnectionPool(DSN, min_size=1, max_size=3, kwargs={"autocommit": True}, open=False)
    await p.open(wait=True)
    async with p.connection() as c:
        await c.execute("TRUNCATE raw.page, raw.payload, raw.archive_manifest, control.lease_loss, control.task")
    try:
        yield p
    finally:
        await p.close()


class Fenced:
    """Stand in for the task's lease: ``fenced`` opens a plain transaction; ``complete`` records the report."""

    def __init__(self, pool: Pool) -> None:
        """Store the pool."""
        self.pool = pool
        self.reports: list[TaskReport] = []

    @asynccontextmanager
    async def fenced(self, conn: Connection) -> AsyncIterator[None]:
        """Run the block in one transaction."""
        async with conn.transaction():
            yield

    async def complete(self, conn: Connection, report: TaskReport) -> str:
        """Record the report."""
        self.reports.append(report)
        return "succeeded"


async def task(conn: Connection, job: str, source: UUID, key: str, state: str = "succeeded") -> int:
    """Insert a task and return its id."""
    cursor = await conn.execute(
        "INSERT INTO control.task (run_id, connector, job, source_id, partition_key, state) "
        "VALUES (1, 'tiktok_shop', %s, %s, %s, %s) RETURNING id",
        (job, source, key, state),
    )
    row = await cursor.fetchone()
    assert row is not None
    return int(row[0])


async def land(conn: Connection, task_id: int, page_no: int, data: Any, minute: int = 0) -> None:
    """Save one page of ``task_id`` whose response is ``{"code": 0, "request_id": ..., "data": data}``."""
    await save_page(
        conn,
        RawPage(
            task_id=task_id,
            page_no=page_no,
            connector="tiktok_shop",
            endpoint="orders_search",
            cursor_in=None if page_no == 0 else f"c{page_no}",
            cursor_next=None,
            fetched_at=datetime(2026, 6, 1, 1, minute, tzinfo=UTC),
            payload={"code": 0, "request_id": f"r{task_id}-{page_no}", "data": data},
        ),
        KEYS,
    )


async def may_orders(pool: Pool) -> dict[str, int]:
    """Land orders pages: two May days for two shops, a June day, and an open May re-fetch; return the task ids."""
    async with pool.connection() as c:
        ids = {
            "b_may_2": await task(c, "orders", SHOP_B, "2026-05-02"),
            "a_may_3": await task(c, "orders", SHOP_A, "2026-05-03"),
            "a_may_1": await task(c, "orders", SHOP_A, "2026-05-01"),
            "a_jun_1": await task(c, "orders", SHOP_A, "2026-06-01"),
        }
        await land(c, ids["b_may_2"], 0, {"orders": [{"id": "b1"}]})
        await land(c, ids["a_may_3"], 0, {"orders": [{"id": "a3"}]})
        await land(c, ids["a_may_1"], 0, {"orders": [{"id": "a1"}]})
        await land(c, ids["a_may_1"], 1, {"orders": [{"id": "a1b"}]}, minute=1)
        await land(c, ids["a_jun_1"], 0, {"orders": [{"id": "j1"}]})
    return ids


def test_object_key_names_the_month_and_job() -> None:
    """Check the key layout: one object per job per month."""
    assert object_key("orders", date(2026, 5, 17)) == "raw/month=2026-05/job=orders.parquet"


async def test_a_month_is_written_read_back_recorded_and_its_pages_marked(pool: Pool, tmp_path: Path) -> None:
    """Check one month end to end: the object's rows, order and responses, the manifest row and the marked pages."""
    ids = await may_orders(pool)
    store = DirectoryStore(tmp_path / "bucket")
    archived = await archive_month(pool, Fenced(pool), store, MonthKey(MAY, "orders"), tmp_path)
    assert (archived.key, archived.rows) == ("raw/month=2026-05/job=orders.parquet", 4)
    frame = pl.read_parquet(tmp_path / "bucket" / archived.key)
    assert frame.select("source_id", "partition_key", "page_no").rows() == [
        (str(SHOP_A), "2026-05-01", 0),
        (str(SHOP_A), "2026-05-01", 1),
        (str(SHOP_A), "2026-05-03", 0),
        (str(SHOP_B), "2026-05-02", 0),
    ]
    first = frame.row(0, named=True)
    assert first["job"] == "orders"
    assert first["request_id"] == f"r{ids['a_may_1']}-0"
    assert pl.Series([first["response"]]).str.json_decode().to_list()[0] == {
        "code": 0,
        "request_id": f"r{ids['a_may_1']}-0",
        "data": {"orders": [{"id": "a1"}]},
    }
    async with pool.connection() as c:
        manifest = await (
            await c.execute("SELECT job, month, object_key, rows, bytes FROM raw.archive_manifest")
        ).fetchall()
        assert manifest == [("orders", MAY, archived.key, 4, archived.size)]
        marked = await (
            await c.execute("SELECT task_id, archived_month FROM raw.page ORDER BY task_id, page_no")
        ).fetchall()
    assert {(t, m) for t, m in marked} == {
        (ids["b_may_2"], MAY),
        (ids["a_may_3"], MAY),
        (ids["a_may_1"], MAY),
        (ids["a_jun_1"], None),
    }
    assert store.head(archived.key) == StoredObject(size=archived.size, sha256=archived.sha256)
    assert not [name for name in os.listdir(tmp_path) if name.startswith("archive-")]  # the local files are gone


async def test_a_month_is_written_once(pool: Pool, tmp_path: Path) -> None:
    """Check a second run finds nothing left to write, and the manifest refuses a second row for the month."""
    await may_orders(pool)
    store = DirectoryStore(tmp_path / "bucket")
    await archive_month(pool, Fenced(pool), store, MonthKey(MAY, "orders"), tmp_path)
    again = await archive_month(pool, Fenced(pool), store, MonthKey(MAY, "orders"), tmp_path)
    assert again.rows == 0
    async with pool.connection() as c:
        with pytest.raises(psycopg.errors.UniqueViolation):
            await c.execute(
                "INSERT INTO raw.archive_manifest (job, month, object_key, rows, bytes, sha256) "
                "VALUES ('orders', '2026-05-01', 'other', 1, 1, sha256('x'))"
            )


class Corrupting(DirectoryStore):
    """A store whose downloads come back with one byte changed."""

    def get(self, key: str, path: Path) -> None:
        """Download, then flip the last byte."""
        super().get(key, path)
        data = bytearray(path.read_bytes())
        data[-1] ^= 0xFF
        path.write_bytes(bytes(data))


async def test_a_failed_read_back_records_nothing(pool: Pool, tmp_path: Path) -> None:
    """Check an object that reads back differently is not recorded and its pages stay unarchived."""
    await may_orders(pool)
    with pytest.raises(ArchiveCheckFailed, match="read back"):
        await archive_month(pool, Fenced(pool), Corrupting(tmp_path / "bucket"), MonthKey(MAY, "orders"), tmp_path)
    async with pool.connection() as c:
        assert await (await c.execute("SELECT count(*) FROM raw.archive_manifest")).fetchone() == (0,)
        assert await (await c.execute("SELECT count(*) FROM raw.page WHERE archived_month IS NOT NULL")).fetchone() == (
            0,
        )


async def test_the_plan_waits_for_settling_and_open_tasks(pool: Pool) -> None:
    """Check which months are ready, waiting, not yet settled, already archived, or of an unknown job."""
    async with pool.connection() as c:
        await land(c, await task(c, "orders", SHOP_A, "2026-05-01"), 0, [])  # 7 days: settled on 10 June, ready
        await land(c, await task(c, "product_day", SHOP_A, "2026-05-01"), 0, [])  # 30 days: not settled yet
        await land(c, await task(c, "returns", SHOP_A, "2026-05-01"), 0, [])  # an open re-fetch: waits
        await task(c, "returns", SHOP_B, "2026-05-09", state="retry_wait")
        await land(c, await task(c, "lives", SHOP_A, "2026-05-01"), 0, [])  # a blocked task: not waited for
        await task(c, "lives", SHOP_B, "2026-05-09", state="blocked")
        await land(c, await task(c, "videos", SHOP_A, "2026-04-30"), 0, [])  # archived: its new pages are re-fetches
        await c.execute(
            "INSERT INTO raw.archive_manifest (job, month, object_key, rows, bytes, sha256) "
            "VALUES ('videos', '2026-04-01', 'k', 1, 1, sha256('x'))"
        )
        await land(c, await task(c, "retired", SHOP_A, "2026-05-01"), 0, [])  # no horizon: never archived
    horizons = {"orders": 7, "product_day": 30, "returns": 1, "lives": 7, "videos": 1}
    plan = await plan_months(pool, horizons, TODAY)
    assert plan.ready == [MonthKey(MAY, "lives"), MonthKey(MAY, "orders")]
    assert plan.waiting == [MonthKey(MAY, "returns")]
    assert plan.unknown == {"retired": 1}
    assert plan.orphans == 0


async def test_check_manifest_finds_missing_and_changed_objects(pool: Pool, tmp_path: Path) -> None:
    """Check the integrity check names an object that is gone and one whose size changed, and passes the rest."""
    store = DirectoryStore(tmp_path / "bucket")
    async with pool.connection() as c:
        for name in ("2026-04", "2026-05", "2026-06"):
            await land(c, await task(c, "orders", SHOP_A, f"{name}-01"), 0, {"m": name})
    for month in (date(2026, 4, 1), MAY, date(2026, 6, 1)):
        await archive_month(pool, Fenced(pool), store, MonthKey(month, "orders"), tmp_path)
    assert await check_manifest(pool, store) == []
    (tmp_path / "bucket" / object_key("orders", MAY)).unlink()
    with (tmp_path / "bucket" / object_key("orders", date(2026, 6, 1))).open("ab") as file:
        file.write(b"x")
    problems = await check_manifest(pool, store)
    assert len(problems) == 2
    assert "month=2026-05" in problems[0] and "not in the store" in problems[0]
    assert "month=2026-06" in problems[1] and "bytes" in problems[1]


def claimed(task_id: int) -> ClaimedTask:
    """Build the claimed archive task."""
    now = datetime.now(UTC)
    return ClaimedTask(
        id=task_id,
        run_id=1,
        connector="pipeline",
        job="archive_raw",
        source_id=UUID(int=0),
        partition_key="none",
        attempt=1,
        lease_token=UUID(int=1),
        created_at=now,
        not_before=now,
        claimed_at=now,
        lease_expires_at=now + timedelta(minutes=5),
        trace_context=None,
    )


async def test_the_job_archives_every_ready_month_and_reports_the_pages(pool: Pool, tmp_path: Path) -> None:
    """Check a run writes each ready month and reports the pages archived."""
    await may_orders(pool)
    lease = Fenced(pool)
    job = ArchiveRawJob(refetch_days={"orders": 2}, store=DirectoryStore(tmp_path / "bucket"), scratch=tmp_path)
    await job(claimed(999), lease)  # Fenced has the parts of a lease the job uses
    assert [(r.outcome, r.rows_written) for r in lease.reports] == [(Outcome.SUCCEEDED, 5)]  # May 4 pages, June 1


async def test_the_job_goes_on_after_a_failed_month_then_fails(pool: Pool, tmp_path: Path) -> None:
    """Check a month that fails its check does not stop the others, and the task then fails naming it."""
    await may_orders(pool)
    store = Corrupting(tmp_path / "bucket")
    job = ArchiveRawJob(refetch_days={"orders": 2}, store=store, scratch=tmp_path)
    lease = Fenced(pool)
    with pytest.raises(ArchiveCheckFailed, match=r"month=2026-05.*month=2026-06"):
        await job(claimed(999), lease)
    assert lease.reports == []


async def test_the_job_needs_a_store(pool: Pool) -> None:
    """Check a worker with no archive store fails the task before reading anything."""
    job = ArchiveRawJob(refetch_days={"orders": 2}, store=None)
    with pytest.raises(ArchiveNotConfigured):
        await job(claimed(999), Fenced(pool))


@pytest.mark.parametrize("days", [-1, 1.5, True])
def test_the_job_refuses_a_bad_horizon(days: Any) -> None:
    """Check a horizon that is not an int of 0 or more is refused."""
    with pytest.raises(ValueError, match="refetch_days"):
        ArchiveRawJob(refetch_days={"orders": days}, store=None)


def drop_ids(response: dict[str, Any]) -> dict[str, Any]:
    """Redact each order's ``id`` to ``"x"``: a stand-in for a job's ``redact_for_archive``."""
    for order in response["data"]["orders"]:
        order["id"] = "x"
    return response


async def test_a_redacted_month_keeps_only_what_the_redaction_returns(pool: Pool, tmp_path: Path) -> None:
    """Check every archived response is the redaction's output, while raw.page keeps the response as fetched."""
    ids = await may_orders(pool)
    store = DirectoryStore(tmp_path / "bucket")
    archived = await archive_month(pool, Fenced(pool), store, MonthKey(MAY, "orders"), tmp_path, drop_ids)
    assert archived.rows == 4
    frame = pl.read_parquet(tmp_path / "bucket" / archived.key)
    responses = frame.get_column("response").str.json_decode().to_list()
    assert {order["id"] for response in responses for order in response["data"]["orders"]} == {"x"}
    assert responses[0] == {"code": 0, "request_id": f"r{ids['a_may_1']}-0", "data": {"orders": [{"id": "x"}]}}
    async with pool.connection() as c:
        kept = await (
            await c.execute(
                "SELECT y.body -> 'data' -> 'orders' -> 0 ->> 'id' FROM raw.page p "
                "JOIN raw.payload y ON y.hash = p.payload_hash WHERE p.task_id = %s AND p.page_no = 0",
                (ids["a_may_1"],),
            )
        ).fetchone()
    assert kept == ("a1",)


async def test_the_job_redacts_only_the_jobs_named(pool: Pool, tmp_path: Path) -> None:
    """Check a run applies a job's redaction to that job's months and archives other jobs as fetched."""
    await may_orders(pool)
    async with pool.connection() as c:
        other = await task(c, "returns", SHOP_A, "2026-05-04")
        await land(c, other, 0, {"orders": [{"id": "r1"}]})
    job = ArchiveRawJob(
        refetch_days={"orders": 2, "returns": 2},
        store=DirectoryStore(tmp_path / "bucket"),
        scratch=tmp_path,
        redactions={"orders": drop_ids},
    )
    await job(claimed(999), Fenced(pool))

    def ids_in(key: str) -> set[str]:
        frame = pl.read_parquet(tmp_path / "bucket" / key)
        return {o["id"] for r in frame.get_column("response").str.json_decode().to_list() for o in r["data"]["orders"]}

    assert ids_in(object_key("orders", MAY)) == {"x"}
    assert ids_in(object_key("returns", MAY)) == {"r1"}


def test_the_job_refuses_a_redaction_for_a_job_with_no_horizon() -> None:
    """Check a redaction must name a job the archive has a horizon for."""
    with pytest.raises(ValueError, match="redacted jobs with no refetch_days: orders"):
        ArchiveRawJob(refetch_days={}, store=None, redactions={"orders": drop_ids})
