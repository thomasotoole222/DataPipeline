"""The extract stage: fetch an endpoint's pages for a claimed task and land each one in ``raw.page``.

Public API:
    extract: Discard a task's landed pages, then fetch and land every page again, returning the records landed.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from pipeline.connectors.client import ConnectorClient
from pipeline.connectors.fetch import Page, fetch_pages
from pipeline.connectors.spec import EndpointSpec
from pipeline.db import raw
from pipeline.db.control import ClaimedTask, Fence
from pipeline.db.pool import Pool
from pipeline.db.raw import RawPage
from pipeline.telemetry import Instruments
from pipeline.timeutil import TimeFrame

#: Read-only empty mapping used as the default ``path_params`` of ``extract``, so no call shares a mutable default.
_NO_PATH_PARAMS: Final[Mapping[str, str]] = MappingProxyType({})

#: Name of the attribute ``extract`` sets on an escaping ``Exception`` to the number of records landed before it.
#: ``worker.loop`` reads it back with ``_count(err, "rows_fetched")``; the two strings must stay equal.
_ROWS_FETCHED: Final = "rows_fetched"


async def extract[C](
    pool: Pool,
    task: ClaimedTask,
    fence: Fence,
    client: ConnectorClient[C],
    spec: EndpointSpec,
    window: str | None,  # a name in spec.windows; None for an endpoint without windows
    frame: TimeFrame | None,  # None exactly when window is None
    instruments: Instruments,
    *,
    path_params: Mapping[str, str] = _NO_PATH_PARAMS,
) -> int:
    """Discard the task's landed pages, then fetch every page of ``spec`` for the task's source and land each one.

    The discard runs in its own transaction fenced by ``fence.fenced``. The fetch always starts at page 0: no
    ``resume_from`` is passed to ``fetch_pages``. Each page is landed by ``_land`` in its own fenced transaction, so
    pages landed before a failure stay committed until the next run of the task discards them. After each page is
    landed, under ``spec.connector`` and ``spec.endpoint``, its ``record_count`` is added to the total and to
    ``instruments.rows`` with attributes ``job`` set to ``task.job`` and ``stage`` set to ``"fetched"``. When an
    ``Exception`` escapes, the attribute named by ``_ROWS_FETCHED`` is set on it to the records landed so far, and it
    is re-raised. ``fetch_pages`` checks ``window``, ``frame`` and ``path_params`` only when the first page is
    requested, so a call they do not suit raises after the discard has committed.

    Preconditions:
        ``raw.page`` exists as ``raw.save_page`` requires. ``fence`` fences on ``task``'s lease. ``spec`` is an
        endpoint of ``task.connector``; ``spec.connector`` is stored with each page as given and is not compared with
        ``task.connector``.

    Args:
        pool: Pool each transaction takes a connection from.
        task: The claimed task; its ``id``, ``source_id`` and ``job`` are used.
        fence: Fences each transaction on the task's lease (the worker's ``Lease``).
        client: The client that sends each request.
        spec: The endpoint to fetch.
        window: The name of one of ``spec.windows``, or ``None`` when ``frame`` is ``None``.
        frame: The span of time to fetch, or ``None`` for an endpoint fetched without a window.
        instruments: Instruments whose ``rows`` counter is incremented per page.
        path_params: Values for the fields in ``spec.path``. Defaults to none.

    Returns:
        The sum of ``record_count`` over the pages landed, 0 or more.

    Raises:
        LeaseConflict: From ``fence.fenced`` when the lease is no longer held, before the discard or a page lands.
        Exception: Whatever ``raw.discard_pages``, ``fetch_pages``, ``_land``, the pool, ``fence.fenced`` or the
            counter raises, unchanged. Every ``Exception`` escapes with ``rows_fetched`` set to the records landed
            before it. A ``BaseException`` that is not an ``Exception``, such as ``asyncio.CancelledError``, escapes
            without it.

    """
    rows = 0
    try:
        async with pool.connection() as conn, fence.fenced(conn):
            await raw.discard_pages(conn, task.id)  # every attempt starts over
        async for page in fetch_pages(client, spec, window, frame, task.source_id, path_params=path_params):
            await _land(pool, task, fence, spec, page)
            rows += page.record_count
            instruments.rows.add(page.record_count, {"job": task.job, "stage": "fetched"})
    except Exception as err:
        setattr(err, _ROWS_FETCHED, rows)
        raise
    return rows


async def _land(pool: Pool, task: ClaimedTask, fence: Fence, spec: EndpointSpec, page: Page) -> None:
    """Save ``page`` as a ``RawPage`` row of ``task`` in ``raw.page``, in one transaction fenced on the task's lease.

    ``record_count`` is not stored. ``RawPage`` re-checks ``page_no`` and ``fetched_at``, which ``Page`` already
    guarantees are 0 or more and aware in UTC.

    Preconditions:
        ``raw.page`` exists as ``raw.save_page`` requires.

    Args:
        pool: Pool the transaction takes a connection from.
        task: The task the page belongs to; its ``id`` is used.
        fence: Fences the transaction on the task's lease.
        spec: The spec that fetched the page; its ``connector`` and ``endpoint`` are stored with the page.
        page: The fetched page to save.

    Returns:
        None.

    Raises:
        LeaseConflict: From ``fence.fenced`` when the lease is no longer held; the page is not saved.
        Exception: Whatever ``RawPage`` validation, the pool, ``fence.fenced`` or ``raw.save_page`` raises, unchanged.
            An existing row with the same ``task_id`` and ``page_no`` is overwritten, not rejected.

    """
    row = RawPage(
        task_id=task.id,
        page_no=page.page_no,
        connector=spec.connector,
        endpoint=spec.endpoint,
        cursor_in=page.cursor_in,
        cursor_next=page.cursor_next,
        fetched_at=page.fetched_at,
        payload=page.payload,
    )
    async with pool.connection() as conn, fence.fenced(conn):  # a connection only for the write, never across a call
        await raw.save_page(conn, row)
