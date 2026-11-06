"""Integration tests for fanned-out jobs: ``stages.extract.fan_out`` and ``BatchJob`` with an id source."""

import asyncio
import json
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime, timedelta
from typing import Annotated, Any, ClassVar
from uuid import UUID
from zoneinfo import ZoneInfo

import psycopg
import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader, NumberDataPoint
from psycopg_pool import AsyncConnectionPool

from pipeline.checks.checks import Validation
from pipeline.checks.record import Record
from pipeline.checks.targets import Snapshot, To, Upsert
from pipeline.connectors.fanout import FromList, FromQuery
from pipeline.connectors.paging import CursorPaginator, NoPaging
from pipeline.connectors.spec import ConnectorRequest, EndpointSpec, IdParam, IdPlace
from pipeline.connectors.windows import UnixSecondsWindow
from pipeline.db import control, raw
from pipeline.db.control import ClaimedTask, TaskReport, claim
from pipeline.db.pool import Connection, Pool
from pipeline.db.raw import RawPage
from pipeline.errors import (
    AuthBlocked,
    ClientErrorUnhandled,
    NoRecord,
    PermanentError,
    RateLimited,
    TransientError,
)
from pipeline.jobs.base import BatchJob
from pipeline.stages.extract import fan_out
from pipeline.telemetry import Instruments
from pipeline.timeutil import TimeFrame

DSN = os.environ.get("TEST_PIPELINE_DSN", "postgresql://postgres@localhost:5432/pipeline_test")
SOURCE = UUID("00000000-0000-0000-0000-000000000001")
OTHER = UUID("00000000-0000-0000-0000-000000000002")
ZONE = ZoneInfo("Europe/London")
FRAME = TimeFrame.days(date(2026, 10, 19), date(2026, 10, 20), ZONE)


def cursor(size: int) -> CursorPaginator:
    """Return a TikTok-style cursor paginator of ``size`` records a page."""
    return CursorPaginator(
        token_param="page_token",  # noqa: S106 - a parameter name, not a password
        size_param="page_size",
        size=size,
        next_at=("data", "next_page_token"),
    )


SEARCH = EndpointSpec(
    name="tiktok_shop.products_search",
    connector="tiktok_shop",
    method="POST",
    path="/product/202502/products/search",
    paginator=cursor(2),
    records_at=("data", "products"),
    windows={"update": UnixSecondsWindow(start_param="update_time_ge", end_param="update_time_lt")},
)
GET_PRODUCT = EndpointSpec(
    name="tiktok_shop.product",
    connector="tiktok_shop",
    method="GET",
    path="/product/202309/products/{product_id}",
    paginator=NoPaging(),
    records_at=("data",),
    id_param=IdParam(place=IdPlace.PATH, name="product_id"),
    one_record=True,
)
MINUTES = EndpointSpec(
    name="tiktok_shop.live_minutes",
    connector="tiktok_shop",
    method="GET",
    path="/live/202508/lives/{live_id}/minutes",
    paginator=cursor(2),
    records_at=("data", "minutes"),
    id_param=IdParam(place=IdPlace.PATH, name="live_id", inject="live_id"),
)
LIVES = EndpointSpec(
    name="tiktok_shop.lives",
    connector="tiktok_shop",
    method="POST",
    path="/live/202508/lives/details",
    paginator=NoPaging(),
    records_at=("data", "lives"),
    id_param=IdParam(place=IdPlace.BODY, name="live_ids", batch=2),
)
LIVES_BY_ID = EndpointSpec(
    name="tiktok_shop.lives",
    connector="tiktok_shop",
    method="POST",
    path="/live/202508/lives/details",
    paginator=NoPaging(),
    records_at=("data", "lives"),
    id_param=IdParam(place=IdPlace.BODY, name="live_ids", batch=3, id_at=("id",)),
)
UNSIZED_MINUTES = EndpointSpec(
    name="tiktok_shop.live_minutes",
    connector="tiktok_shop",
    method="GET",
    path="/live/202508/lives/{live_id}/minutes",
    paginator=CursorPaginator(
        token_param="page_token",  # noqa: S106 - a parameter name, not a password
        size_param=None,
        size=None,
        next_at=("data", "next_page_token"),
    ),
    records_at=("data", "minutes"),
    id_param=IdParam(place=IdPlace.PATH, name="live_id", inject="live_id"),
)
REPORT = EndpointSpec(
    name="tiktok_shop.gmv_max_report",
    connector="tiktok_shop",
    method="GET",
    path="/open_api/v1.3/gmv_max/report/get/",
    paginator=NoPaging(),
    records_at=("data", "list"),
    id_param=IdParam(place=IdPlace.QUERY, name="filtering", batch=2, encoding="json", wrap=("campaign_ids",)),
)
LIVE_IDS = FromQuery(
    "SELECT live_id FROM t_fan.live_source WHERE source_id = %(source_id)s "
    "AND started_at >= %(frame_start)s AND started_at < %(frame_end)s"
)


class Product(Record, table="t_fan.product", writer=Upsert(("source_id", "product_id"))):
    """A product: Get Product's record, or the search row Get Product left out."""

    record_key: ClassVar[tuple[str, ...]] = ("id",)
    record_version: ClassVar[int] = 2

    id: Annotated[str, To("product_id")]
    status: Annotated[str, To("status")]
    title: Annotated[str | None, To("title")] = None  # only Get Product returns it


class Minute(Record, table="t_fan.minute", writer=Upsert(("source_id", "live_id", "minute"))):
    """One minute of a LIVE, which does not carry its live_id."""

    record_key: ClassVar[tuple[str, ...]] = ("live_id", "minute")
    record_version: ClassVar[int] = 1

    live_id: Annotated[str, To("live_id")]
    minute: Annotated[int, To("minute")]
    viewers: Annotated[int, To("viewers")]


class MinuteSnapshot(Record, table="t_fan.minute_snapshot", writer=Snapshot(("source_id", "live_id", "minute"))):
    """A LIVE minute in a table pruned to what each task loads."""

    record_key: ClassVar[tuple[str, ...]] = ("live_id", "minute")
    record_version: ClassVar[int] = 1

    live_id: Annotated[str, To("live_id")]
    minute: Annotated[int, To("minute")]


class Live(Record, table="t_fan.live", writer=Upsert(("source_id", "live_id"))):
    """A LIVE's totals, fetched in batches."""

    record_key: ClassVar[tuple[str, ...]] = ("id",)
    record_version: ClassVar[int] = 1

    id: Annotated[str, To("live_id")]
    gmv: Annotated[int, To("gmv")]


class Products(BatchJob[Product]):
    """Search products updated in the day, then Get Product for each, keeping the row of one it leaves out."""

    name = "products"
    spec = GET_PRODUCT
    window = "update"
    partition = "day"
    validation = Validation(model=Product)
    id_source = FromList(spec=SEARCH, id_at=("id",), keep_list_row=True)


class Minutes(BatchJob[Minute]):
    """Fetch each of the day's LIVEs' per-minute rows, paged, the live_id written into each."""

    name = "live_minutes"
    spec = MINUTES
    window = "update"
    partition = "day"
    validation = Validation(model=Minute)
    id_source = LIVE_IDS


class MinuteSnapshots(BatchJob[MinuteSnapshot]):
    """The day's LIVEs' minutes, loaded into a pruned table."""

    name = "live_minutes"
    spec = MINUTES
    window = "update"
    partition = "day"
    validation = Validation(model=MinuteSnapshot)
    id_source = LIVE_IDS


class Lives(BatchJob[Live]):
    """Fetch the day's LIVEs' totals, two LIVEs a call."""

    name = "lives"
    spec = LIVES
    window = "update"
    partition = "day"
    validation = Validation(model=Live)
    id_source = LIVE_IDS
    concurrency = 1


class LivesById(BatchJob[Live]):
    """Fetch the day's LIVEs' totals, three LIVEs a call, each record read for the id it is for."""

    name = "lives"
    spec = LIVES_BY_ID
    window = "update"
    partition = "day"
    validation = Validation(model=Live)
    id_source = LIVE_IDS
    concurrency = 1


class UnsizedMinutes(BatchJob[Minute]):
    """Fetch each of the day's LIVEs' per-minute rows from an endpoint that takes no page size."""

    name = "live_minutes"
    spec = UNSIZED_MINUTES
    window = "update"
    partition = "day"
    validation = Validation(model=Minute)
    id_source = LIVE_IDS


class Reports(BatchJob[Live]):
    """Fetch the day's campaigns' totals from a GET report, two campaign ids a call in a JSON filter."""

    name = "campaign_report"
    spec = REPORT
    window = "update"
    partition = "day"
    validation = Validation(model=Live)
    id_source = LIVE_IDS
    concurrency = 1


type Reply = Callable[[ConnectorRequest], dict[str, Any] | Exception]
type Hook = Callable[[ConnectorRequest], Awaitable[None]]


class FakeClient:
    """A connector client that answers each request with ``reply`` after running a hook and yielding once."""

    connector = "tiktok_shop"
    hashed_keys = frozenset({"data"})

    def __init__(self, reply: Reply, hook: Hook | None = None) -> None:
        """Answer with ``reply``; an exception it returns is raised instead."""
        self.reply = reply
        self.hook = hook
        self.requests: list[ConnectorRequest] = []

    async def send(self, request: ConnectorRequest, source_id: UUID) -> dict[str, Any]:
        """Record the request, run the hook, let other calls run, then answer."""
        assert source_id == SOURCE
        self.requests.append(request)
        if self.hook is not None:
            await self.hook(request)
        await asyncio.sleep(0)  # concurrent calls interleave
        answer = self.reply(request)
        if isinstance(answer, Exception):
            raise answer
        return answer


def products(
    search: list[list[dict[str, Any]]], details: Mapping[str, dict[str, Any] | Exception]
) -> Callable[[ConnectorRequest], dict[str, Any] | Exception]:
    """Answer a product search with ``search``'s pages and Get Product from ``details`` by product id."""

    def reply(request: ConnectorRequest) -> dict[str, Any] | Exception:
        if request.path == SEARCH.path:
            at = int(request.query.get("page_token", "0"))
            following = str(at + 1) if at + 1 < len(search) else ""
            return {"code": 0, "data": {"products": search[at], "next_page_token": following}}
        return details[request.path.rsplit("/", 1)[1]]

    return reply


def product(product_id: str, title: str) -> dict[str, Any]:
    """Return a Get Product body."""
    return {"code": 0, "data": {"id": product_id, "status": "ACTIVATE", "title": title}}


def row(product_id: str, status: str = "ACTIVATE") -> dict[str, Any]:
    """Return a product search row."""
    return {"id": product_id, "status": status}


@pytest.fixture
async def pool() -> AsyncIterator[Pool]:
    """Yield an open pool on empty control, raw and quarantine tables and fresh test tables."""
    p: Pool = AsyncConnectionPool(DSN, min_size=1, max_size=2, kwargs={"autocommit": True}, open=False)
    await p.open(wait=True)
    async with p.connection() as c:
        await c.execute("TRUNCATE control.lease_loss, control.task RESTART IDENTITY")
        await c.execute("TRUNCATE raw.page, raw.payload, quarantine.row")
        await c.execute("DROP SCHEMA IF EXISTS t_fan CASCADE")
        await c.execute("CREATE SCHEMA t_fan")
        await c.execute(
            "CREATE TABLE t_fan.product (source_id uuid NOT NULL, product_id text NOT NULL, status text NOT NULL, "
            "title text, PRIMARY KEY (source_id, product_id))"
        )
        await c.execute(
            "CREATE TABLE t_fan.minute (source_id uuid NOT NULL, live_id text NOT NULL, minute bigint NOT NULL, "
            "viewers bigint NOT NULL, PRIMARY KEY (source_id, live_id, minute))"
        )
        await c.execute(
            "CREATE TABLE t_fan.minute_snapshot (source_id uuid NOT NULL, live_id text NOT NULL, "
            "minute bigint NOT NULL, removed_at timestamptz, PRIMARY KEY (source_id, live_id, minute))"
        )
        await c.execute(
            "CREATE TABLE t_fan.live (source_id uuid NOT NULL, live_id text NOT NULL, gmv bigint NOT NULL, "
            "PRIMARY KEY (source_id, live_id))"
        )
        await c.execute("CREATE TABLE t_fan.live_source (source_id uuid, live_id text, started_at timestamptz)")
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
        check_failures=meter.create_counter("pipeline.check.failures"),
        text_replaced=meter.create_counter("pipeline.connector.text_replaced"),
    )


def stage_rows(reader: InMemoryMetricReader) -> dict[str, int]:
    """Return the ``pipeline.rows`` totals by stage."""
    data = reader.get_metrics_data()
    totals: dict[str, int] = {}
    if data is None:
        return totals
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name != "pipeline.rows":
                    continue
                for point in metric.data.data_points:
                    if isinstance(point, NumberDataPoint) and "table" not in (point.attributes or {}):
                        stage = str((point.attributes or {})["stage"])
                        totals[stage] = totals.get(stage, 0) + int(point.value)
    return totals


class FakeFence:
    """A ``Fence`` that opens a transaction and runs ``control.fence`` in it, as the worker's ``Lease.fenced`` does."""

    def __init__(self, task: ClaimedTask) -> None:
        """Fence on ``task``."""
        self.task = task

    @asynccontextmanager
    async def fenced(self, conn: Connection) -> AsyncIterator[None]:
        """Open a transaction on ``conn``, fence it on the task's lease, and run the block."""
        async with conn.transaction():
            await control.fence(conn, self.task)
            yield


class FakeLease(FakeFence):
    """A ``TaskLease`` on a pool that fences like ``FakeFence`` and reports through ``control.report_outcome``."""

    def __init__(self, task: ClaimedTask, pool: Pool) -> None:
        """Lease ``task`` on ``pool``."""
        super().__init__(task)
        self._pool = pool

    @property
    def pool(self) -> Pool:
        """Return the pool."""
        return self._pool

    async def complete(self, conn: Connection, report: TaskReport) -> str:
        """Report the outcome."""
        return await control.report_outcome(conn, self.task, report)


async def claimed(pool: Pool, job: str) -> ClaimedTask:
    """Insert a pending task of ``job`` for 2026-10-19 and claim it."""
    async with pool.connection() as c:
        await c.execute(
            "INSERT INTO control.task (run_id, connector, job, source_id, partition_key) "
            "VALUES (7, 'tiktok_shop', %s, %s, '2026-10-19')",
            (job, SOURCE),
        )
        task = await claim(c, "test-worker")
    assert task is not None
    return task


async def run(pool: Pool, job: type[BatchJob[Any]], client: FakeClient, instruments: Instruments) -> ClaimedTask:
    """Claim a task of ``job`` and run it."""
    task = await claimed(pool, job.name)
    await job(client=client, zone=ZONE, instruments=instruments)(task, FakeLease(task, pool))
    return task


async def rows(pool: Pool, query: str) -> list[tuple[Any, ...]]:
    """Return every row ``query`` selects."""
    async with pool.connection() as c:
        cursor = await c.execute(query)
        return await cursor.fetchall()


async def pages(pool: Pool, task_id: int) -> list[RawPage]:
    """Read a task's landed pages."""
    async with pool.connection() as c:
        return [p async for p in raw.read_pages(c, task_id)]


async def lives(pool: Pool, *started: tuple[UUID, str, datetime]) -> None:
    """Record LIVEs for the id query: (source, live id, start)."""
    async with pool.connection() as c:
        for source_id, live_id, at in started:
            await c.execute("INSERT INTO t_fan.live_source VALUES (%s, %s, %s)", (source_id, live_id, at))


def minutes(pages_by_live: dict[str, list[list[tuple[int, int]]]]) -> Reply:
    """Answer the per-minute endpoint with each LIVE's pages of (minute, viewers)."""

    def reply(request: ConnectorRequest) -> dict[str, Any] | Exception:
        live_id = request.path.split("/")[-2]
        at = int(request.query.get("page_token", "0"))
        live_pages = pages_by_live[live_id]
        following = str(at + 1) if at + 1 < len(live_pages) else ""
        found = [{"minute": m, "viewers": v} for m, v in live_pages[at]]
        return {"code": 0, "data": {"minutes": found, "next_page_token": following}}

    return reply


NOON = datetime(2026, 10, 19, 12, tzinfo=UTC)


async def test_a_list_fans_out_per_id_and_keeps_the_row_of_an_omitted_id(
    pool: Pool, instruments: Instruments, reader: InMemoryMetricReader
) -> None:
    """Check the list is drained, each id called once in id order, and an omitted product loads its list row."""
    search = [[row("p3"), row("p1")], [row("p2", "FREEZE"), row("p1")]]  # p1 listed twice: called once
    client = FakeClient(products(search, {"p1": product("p1", "One"), "p2": {"code": 0}, "p3": product("p3", "Three")}))
    task = await run(pool, Products, client, instruments)
    assert await rows(pool, "SELECT product_id, status, title FROM t_fan.product ORDER BY product_id") == [
        ("p1", "ACTIVATE", "One"),
        ("p2", "FREEZE", None),
        ("p3", "ACTIVATE", "Three"),
    ]
    landed = await pages(pool, task.id)
    assert [(p.page_no, p.endpoint, p.ids, p.error_class) for p in landed] == [
        (0, "products_search", None, None),
        (1, "products_search", None, None),
        (2, "product", ("p1",), None),
        (3, "product", ("p2",), None),
        (4, "product", ("p3",), None),
    ]
    searches = [r for r in client.requests if r.path == SEARCH.path]
    assert all(r.body is not None and r.body["update_time_ge"] == int(FRAME.start.timestamp()) for r in searches)
    gets = sorted(r.path for r in client.requests if r.path != SEARCH.path)
    assert gets == [f"/product/202309/products/{p}" for p in ("p1", "p2", "p3")]
    assert all(r.body is None and not r.query for r in client.requests if r.path != SEARCH.path)  # no window
    assert await rows(pool, "SELECT state, rows_fetched, rows_written, rows_rejected FROM control.task") == [
        ("succeeded", 6, 3, 0)
    ]
    assert stage_rows(reader) == {"fetched": 4, "fanout_fetched": 2}


async def test_a_per_id_call_pages_and_writes_its_id_into_each_record(pool: Pool, instruments: Instruments) -> None:
    """Check a paged per-id call lands every page under its ids, and each record gets its live_id."""
    await lives(pool, (SOURCE, "L2", NOON), (SOURCE, "L1", NOON))
    client = FakeClient(minutes({"L1": [[(1, 10), (2, 20)], [(3, 30)]], "L2": [[(1, 5)]]}))
    task = await run(pool, Minutes, client, instruments)
    assert await rows(pool, "SELECT live_id, minute, viewers FROM t_fan.minute ORDER BY live_id, minute") == [
        ("L1", 1, 10),
        ("L1", 2, 20),
        ("L1", 3, 30),
        ("L2", 1, 5),
    ]
    landed = await pages(pool, task.id)
    assert [(p.page_no, p.ids) for p in landed] == [(0, ("L1",)), (1, ("L1",)), (2, ("L2",))]
    assert landed[0].cursor_next == landed[1].cursor_in
    assert landed[1].cursor_next is None and landed[2].cursor_in is None
    assert "live_id" not in landed[0].payload["data"]["minutes"][0]  # raw stays as fetched


async def test_query_ids_are_the_tasks_source_and_frame(pool: Pool, instruments: Instruments) -> None:
    """Check the id query is bound to the task's source and frame, and each distinct id is called once."""
    await lives(
        pool,
        (SOURCE, "L1", NOON),
        (SOURCE, "L1", NOON + timedelta(hours=1)),  # the same LIVE twice: one call
        (SOURCE, "L9", NOON + timedelta(days=1)),  # outside the frame
        (OTHER, "L5", NOON),  # another source's
    )
    client = FakeClient(minutes({"L1": [[(1, 10)]]}))
    await run(pool, Minutes, client, instruments)
    assert [r.path for r in client.requests] == ["/live/202508/lives/L1/minutes"]


async def test_body_ids_are_sent_in_batches(pool: Pool, instruments: Instruments) -> None:
    """Check body ids go in batches of up to ``batch``, in id order, each page landed under its call's ids."""
    await lives(pool, (SOURCE, "c", NOON), (SOURCE, "a", NOON), (SOURCE, "b", NOON))

    def reply(request: ConnectorRequest) -> dict[str, Any]:
        assert request.body is not None
        return {"code": 0, "data": {"lives": [{"id": i, "gmv": ord(i)} for i in request.body["live_ids"]]}}

    client = FakeClient(reply)
    task = await run(pool, Lives, client, instruments)
    assert [r.body for r in client.requests] == [{"live_ids": ["a", "b"]}, {"live_ids": ["c"]}]
    assert [p.ids for p in await pages(pool, task.id)] == [("a", "b"), ("c",)]
    assert await rows(pool, "SELECT live_id, gmv FROM t_fan.live ORDER BY live_id") == [
        ("a", 97),
        ("b", 98),
        ("c", 99),
    ]


@pytest.mark.parametrize(
    "failure",
    [
        GET_PRODUCT.unhandled("no product at data"),
        PermanentError("refused"),
        TransientError("timed out after retries"),
        RateLimited("slow down", retry_after=timedelta(seconds=30)),
    ],
)
async def test_a_failed_call_is_held_as_detail_failed_while_the_others_load(
    pool: Pool, instruments: Instruments, reader: InMemoryMetricReader, failure: Exception
) -> None:
    """Check a holdable failure lands a marker, holds the id's list row under detail_failed, and the rest loads."""
    search = [[row("p1"), row("p2"), row("p3")]]
    details: dict[str, dict[str, Any] | Exception] = {
        "p1": product("p1", "One"),
        "p2": failure,
        "p3": product("p3", "Three"),
    }
    task = await run(pool, Products, FakeClient(products(search, details)), instruments)
    assert await rows(pool, "SELECT product_id FROM t_fan.product ORDER BY product_id") == [("p1",), ("p3",)]
    marker = (await pages(pool, task.id))[2]
    error_class = type(failure).__name__
    assert (marker.ids, marker.error_class, marker.payload) == (("p2",), error_class, {"error": str(failure)})
    assert await rows(pool, "SELECT raw_page_no, group_key, payload, failures FROM quarantine.row") == [
        (
            2,
            '["p2"]',
            row("p2"),
            [{"rule": "detail_failed", "version": 2, "message": f"{error_class}: {failure}", "until": None}],
        )
    ]
    assert await rows(pool, "SELECT state, rows_rejected FROM control.task") == [("succeeded", 1)]
    assert stage_rows(reader)["fanout_failed"] == 1


@pytest.mark.parametrize(
    "failure",
    [
        AuthBlocked(connector="tiktok_shop", reason="token revoked"),
        RateLimited("slow down", retry_after=timedelta(minutes=10)),
    ],
)
async def test_a_blocking_failure_fails_the_task(pool: Pool, instruments: Instruments, failure: Exception) -> None:
    """Check AuthBlocked or a long RateLimited fails the task with the rows fetched so far, and nothing loads."""
    search = [[row("p1"), row("p2"), row("p3")]]
    details: dict[str, dict[str, Any] | Exception] = {
        "p1": product("p1", "One"),
        "p2": failure,
        "p3": product("p3", "Three"),
    }
    task = await claimed(pool, "products")
    job = Products(client=FakeClient(products(search, details)), zone=ZONE, instruments=instruments)
    with pytest.raises(type(failure)) as caught:
        await job(task, FakeLease(task, pool))
    assert caught.value is failure
    assert caught.value.rows_fetched == 4  # type: ignore[attr-defined]  # three listed, p1 landed before p2
    assert await rows(pool, "SELECT product_id FROM t_fan.product") == []
    assert all(p.error_class is None for p in await pages(pool, task.id))


async def test_an_empty_id_query_succeeds_with_nothing_fetched_or_pruned(pool: Pool, instruments: Instruments) -> None:
    """Check a day with no ids makes no call, prunes nothing and succeeds, even into a Snapshot table."""
    async with pool.connection() as c:
        await c.execute("INSERT INTO t_fan.minute_snapshot VALUES (%s, 'L0', 1, NULL)", (SOURCE,))
    client = FakeClient(minutes({}))
    await run(pool, MinuteSnapshots, client, instruments)
    assert client.requests == []
    assert await rows(pool, "SELECT live_id, removed_at FROM t_fan.minute_snapshot") == [("L0", None)]
    assert await rows(pool, "SELECT state, rows_fetched, rows_written, rows_rejected FROM control.task") == [
        ("succeeded", 0, 0, 0)
    ]


async def test_an_empty_list_succeeds_with_nothing_called(pool: Pool, instruments: Instruments) -> None:
    """Check an empty list page lands, no per-id call is made, and the task succeeds."""
    client = FakeClient(products([[]], {}))
    task = await run(pool, Products, client, instruments)
    assert [r.path for r in client.requests] == [SEARCH.path]
    assert [p.endpoint for p in await pages(pool, task.id)] == ["products_search"]
    assert await rows(pool, "SELECT state FROM control.task") == [("succeeded",)]


async def test_no_connection_is_held_across_a_fanned_out_call(instruments: Instruments, pool: Pool) -> None:
    """Check a one-connection pool is free whenever the connector is called, list and per-id calls alike."""
    single: Pool = AsyncConnectionPool(DSN, min_size=1, max_size=1, kwargs={"autocommit": True}, open=False)
    await single.open(wait=True)
    try:
        task = await claimed(single, "products")

        async def borrow(request: ConnectorRequest) -> None:
            async with single.connection(timeout=0.5):
                pass

        search = [[row("p1"), row("p2")], [row("p3")]]
        details = {p: product(p, p) for p in ("p1", "p2", "p3")}
        client = FakeClient(products(search, details), borrow)
        await Products(client=client, zone=ZONE, instruments=instruments)(task, FakeLease(task, single))
        assert len(client.requests) == 5
    finally:
        await single.close()


async def test_a_retry_starts_over_and_repeats_every_call(pool: Pool, instruments: Instruments) -> None:
    """Check a second attempt discards the first's pages and makes every call again."""
    task = await claimed(pool, "products")
    fence = FakeFence(task)
    source = Products.id_source
    assert source is not None
    first = FakeClient(products([[row("p1"), row("p2")]], {"p1": product("p1", "1"), "p2": product("p2", "2")}))
    await fan_out(pool, task, fence, first, GET_PRODUCT, "update", FRAME, instruments, source=source, concurrency=5)
    again = FakeClient(products([[row("p2")]], {"p2": product("p2", "two")}))
    fanned = await fan_out(
        pool, task, fence, again, GET_PRODUCT, "update", FRAME, instruments, source=source, concurrency=5
    )
    assert (fanned.rows, fanned.ids) == (2, 1)
    assert [r.path for r in again.requests] == [SEARCH.path, "/product/202309/products/p2"]
    landed = await pages(pool, task.id)
    assert [(p.page_no, p.ids) for p in landed] == [(0, None), (1, ("p2",))]
    assert landed[1].payload == product("p2", "two")


async def test_a_list_row_without_an_id_fails_the_task(pool: Pool, instruments: Instruments) -> None:
    """Check a list record whose id cannot be read fails as a response the list spec cannot read."""
    client = FakeClient(products([[{"status": "ACTIVATE"}]], {}))
    task = await claimed(pool, "products")
    with pytest.raises(ClientErrorUnhandled, match="no id at id"):
        await Products(client=client, zone=ZONE, instruments=instruments)(task, FakeLease(task, pool))


class Writes(BatchJob[Minute]):
    """A job whose id query tries to write."""

    name = "live_minutes"
    spec = MINUTES
    window = "update"
    partition = "day"
    validation = Validation(model=Minute)
    id_source = FromQuery("INSERT INTO t_fan.live_source (live_id) VALUES ('x') RETURNING live_id")


async def test_the_id_query_runs_read_only(pool: Pool, instruments: Instruments) -> None:
    """Check an id query that writes fails, before any call, and writes nothing."""
    client = FakeClient(minutes({}))
    with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
        await run(pool, Writes, client, instruments)
    assert client.requests == []
    assert await rows(pool, "SELECT live_id FROM t_fan.live_source") == []


class Smuggles(BatchJob[Minute]):
    """A job whose parameterless id query hides a DELETE after its SELECT."""

    name = "live_minutes"
    spec = MINUTES
    window = "update"
    partition = "day"
    validation = Validation(model=Minute)
    id_source = FromQuery("SELECT live_id FROM t_fan.live_source; DELETE FROM t_fan.live_source")


async def test_an_id_query_of_several_statements_is_refused(pool: Pool, instruments: Instruments) -> None:
    """Check a multi-statement query with no parameter is refused outright, and its DELETE never runs."""
    await lives(pool, (SOURCE, "L1", NOON))
    client = FakeClient(minutes({}))
    with pytest.raises(psycopg.errors.SyntaxError, match="cannot insert multiple commands"):
        await run(pool, Smuggles, client, instruments)
    assert client.requests == []
    assert await rows(pool, "SELECT live_id FROM t_fan.live_source") == [("L1",)]


async def test_a_marker_lands_when_the_error_message_holds_a_nul(pool: Pool, instruments: Instruments) -> None:
    """Check a NUL in a held failure's message is replaced, so the marker lands and the others load."""
    search = [[row("p1"), row("p2")]]
    details: dict[str, dict[str, Any] | Exception] = {"p1": product("p1", "One"), "p2": PermanentError("bad\x00body")}
    task = await run(pool, Products, FakeClient(products(search, details)), instruments)
    marker = (await pages(pool, task.id))[2]
    assert (marker.error_class, marker.payload) == ("PermanentError", {"error": "bad�body"})
    ((failures,),) = await rows(pool, "SELECT failures FROM quarantine.row")
    assert failures[0]["message"] == "PermanentError: bad�body"
    assert await rows(pool, "SELECT product_id FROM t_fan.product") == [("p1",)]
    assert await rows(pool, "SELECT state FROM control.task") == [("succeeded",)]


async def test_an_id_the_path_cannot_take_is_held_as_detail_failed(pool: Pool, instruments: Instruments) -> None:
    """Check a list id of ".." lands a marker and is held, unsent, while the other ids load."""
    search = [[row("p1"), row(".."), row("p3")]]
    details: dict[str, dict[str, Any] | Exception] = {"p1": product("p1", "One"), "p3": product("p3", "Three")}
    client = FakeClient(products(search, details))
    task = await run(pool, Products, client, instruments)
    assert sorted(r.path for r in client.requests if r.path != SEARCH.path) == [
        "/product/202309/products/p1",
        "/product/202309/products/p3",
    ]
    marker = (await pages(pool, task.id))[1]  # ".." sorts first among the ids
    assert (marker.ids, marker.error_class) == (("..",), "IdRefused")
    ((payload, failures),) = await rows(pool, "SELECT payload, failures FROM quarantine.row")
    assert payload == row("..")
    assert failures[0]["rule"] == "detail_failed"
    assert failures[0]["message"].startswith("IdRefused: tiktok_shop.product: dot path values")
    assert await rows(pool, "SELECT product_id FROM t_fan.product ORDER BY product_id") == [("p1",), ("p3",)]


GET_PRODUCT_INJECTED = EndpointSpec(
    name="tiktok_shop.product",
    connector="tiktok_shop",
    method="GET",
    path="/product/202309/products/{product_id}",
    paginator=NoPaging(),
    records_at=("data",),
    id_param=IdParam(place=IdPlace.PATH, name="product_id", inject="product_id"),
    one_record=True,
)


class KeyedProduct(Record, table="t_fan.product", writer=Upsert(("source_id", "product_id"))):
    """A product keyed by the injected product_id, which neither the search row nor Get Product carries."""

    record_key: ClassVar[tuple[str, ...]] = ("product_id",)
    record_version: ClassVar[int] = 1

    product_id: Annotated[str, To("product_id")]
    status: Annotated[str, To("status")]
    title: Annotated[str | None, To("title")] = None


class KeyedProducts(BatchJob[KeyedProduct]):
    """Products keyed by the injected id, keeping the row of a product Get Product leaves out."""

    name = "products"
    spec = GET_PRODUCT_INJECTED
    window = "update"
    partition = "day"
    validation = Validation(model=KeyedProduct)
    id_source = FromList(spec=SEARCH, id_at=("ref",), keep_list_row=True)


async def test_a_kept_list_row_gets_the_injected_id(pool: Pool, instruments: Instruments) -> None:
    """Check a kept list row is given the injected id too, so it loads under its key like a fetched record."""
    search = [[{"ref": "p1", "status": "ACTIVATE"}, {"ref": "p2", "status": "FREEZE"}]]
    details: dict[str, dict[str, Any] | Exception] = {
        "p1": {"code": 0, "data": {"status": "ACTIVATE", "title": "One"}},
        "p2": {"code": 0},
    }
    task = await run(pool, KeyedProducts, FakeClient(products(search, details)), instruments)
    assert await rows(pool, "SELECT product_id, status, title FROM t_fan.product ORDER BY product_id") == [
        ("p1", "ACTIVATE", "One"),
        ("p2", "FREEZE", None),
    ]
    assert await rows(pool, "SELECT count(*) FROM quarantine.row") == [(0,)]
    assert "product_id" not in (await pages(pool, task.id))[0].payload["data"]["products"][1]  # raw unchanged


@pytest.mark.parametrize(
    ("ids", "error_class"),
    [
        ("{a,NULL}", None),
        ("{{a},{b}}", None),
        ("{a}", ""),
    ],
)
async def test_raw_page_refuses_a_null_or_nested_id_and_an_empty_error_class(
    pool: Pool, ids: str, error_class: str | None
) -> None:
    """Check raw.page's CHECKs refuse a NULL id, a two-dimensional ids array and an error_class of ''."""
    async with pool.connection() as c:
        await c.execute("INSERT INTO raw.payload VALUES ('\\x00', '{}')")
        with pytest.raises(psycopg.errors.CheckViolation):
            await c.execute(
                "INSERT INTO raw.page (task_id, page_no, connector, endpoint, fetched_at, envelope, payload_hash, "
                "ids, error_class) VALUES (1, 0, 'c', 'e', now(), '{}', '\\x00', %s::text[], %s)",
                (ids, error_class),
            )


async def test_a_no_record_answer_or_an_empty_object_loads_the_list_row(pool: Pool, instruments: Instruments) -> None:
    """Check a call answered with NoRecord lands an empty page, and it and a data of {} both keep the list row."""
    search = [[row("p1"), row("p2", "FREEZE"), row("p3", "DELETED")]]
    details: dict[str, dict[str, Any] | Exception] = {
        "p1": product("p1", "One"),
        "p2": NoRecord("product frozen"),
        "p3": {"code": 0, "data": {}},
    }
    task = await run(pool, Products, FakeClient(products(search, details)), instruments)
    assert await rows(pool, "SELECT product_id, status, title FROM t_fan.product ORDER BY product_id") == [
        ("p1", "ACTIVATE", "One"),
        ("p2", "FREEZE", None),
        ("p3", "DELETED", None),
    ]
    landed = await pages(pool, task.id)
    assert [(p.ids, p.payload, p.error_class) for p in landed[1:]] == [
        (("p1",), product("p1", "One"), None),
        (("p2",), {}, None),
        (("p3",), {"code": 0, "data": {}}, None),
    ]
    assert await rows(pool, "SELECT count(*) FROM quarantine.row") == [(0,)]


async def test_a_no_record_answer_on_a_later_page_is_held_as_detail_failed(
    pool: Pool, instruments: Instruments
) -> None:
    """Check NoRecord after a call's first page fails the call, held as detail_failed, while the others load."""
    await lives(pool, (SOURCE, "L1", NOON), (SOURCE, "L2", NOON))
    paged = minutes({"L1": [[(1, 10), (2, 20)], [(3, 30)]], "L2": [[(1, 5)]]})

    def reply(request: ConnectorRequest) -> dict[str, Any] | Exception:
        later = request.path.split("/")[-2] == "L1" and "page_token" in request.query
        return NoRecord("room gone") if later else paged(request)

    await run(pool, Minutes, FakeClient(reply), instruments)
    assert await rows(pool, "SELECT live_id, minute FROM t_fan.minute") == [("L2", 1)]
    held = await rows(pool, "SELECT payload, failures->0->>'rule' FROM quarantine.row")
    assert held == [({"id": "L1"}, "detail_failed")]


async def test_query_ids_are_sent_in_json_batches_inside_their_filter(pool: Pool, instruments: Instruments) -> None:
    """Check a GET's query ids go in batches as filtering={"campaign_ids":[...]}, each page under its call's ids."""
    await lives(pool, (SOURCE, "c", NOON), (SOURCE, "a", NOON), (SOURCE, "b", NOON))

    def reply(request: ConnectorRequest) -> dict[str, Any]:
        found = json.loads(request.query["filtering"])["campaign_ids"]
        return {"code": 0, "data": {"list": [{"id": i, "gmv": ord(i)} for i in found]}}

    client = FakeClient(reply)
    task = await run(pool, Reports, client, instruments)
    assert [r.query for r in client.requests] == [
        {"filtering": '{"campaign_ids":["a","b"]}'},
        {"filtering": '{"campaign_ids":["c"]}'},
    ]
    assert [p.ids for p in await pages(pool, task.id)] == [("a", "b"), ("c",)]
    assert await rows(pool, "SELECT live_id, gmv FROM t_fan.live ORDER BY live_id") == [("a", 97), ("b", 98), ("c", 99)]


async def test_an_endpoint_without_a_page_size_is_paged_by_its_token_alone(
    pool: Pool, instruments: Instruments
) -> None:
    """Check a cursor paginator with no size_param sends no page size, only the token after the first page."""
    await lives(pool, (SOURCE, "L1", NOON))
    client = FakeClient(minutes({"L1": [[(1, 10), (2, 20)], [(3, 30)]]}))
    await run(pool, UnsizedMinutes, client, instruments)
    assert [r.query for r in client.requests] == [{}, {"page_token": "1"}]
    assert await rows(pool, "SELECT live_id, minute, viewers FROM t_fan.minute ORDER BY minute") == [
        ("L1", 1, 10),
        ("L1", 2, 20),
        ("L1", 3, 30),
    ]


@pytest.mark.parametrize(("job", "held"), [(LivesById, 1), (Lives, 0)], ids=["id_at", "no id_at"])
async def test_an_id_a_batch_call_leaves_out_is_held_as_detail_failed_with_id_at(
    pool: Pool, instruments: Instruments, job: type[BatchJob[Live]], held: int
) -> None:
    """Check a batch of three ids answered with two records loads both and, with id_at, holds the third.

    Without id_at the left-out id is not noticed: nothing is held for it.
    """
    await lives(pool, (SOURCE, "a", NOON), (SOURCE, "b", NOON), (SOURCE, "c", NOON))

    def reply(request: ConnectorRequest) -> dict[str, Any]:
        assert request.body is not None
        return {"code": 0, "data": {"lives": [{"id": i, "gmv": ord(i)} for i in request.body["live_ids"] if i != "b"]}}

    await run(pool, job, FakeClient(reply), instruments)
    assert await rows(pool, "SELECT live_id FROM t_fan.live ORDER BY live_id") == [("a",), ("c",)]
    expected = [
        (
            0,
            1,
            '["b"]',
            {"id": "b"},
            [{"rule": "detail_failed", "version": 1, "message": "no record returned for the id", "until": None}],
        )
    ]
    assert (
        await rows(pool, "SELECT raw_page_no, raw_index, group_key, payload, failures FROM quarantine.row")
        == expected[:held]
    )
    assert await rows(pool, "SELECT state, rows_rejected FROM control.task") == [("succeeded", held)]
