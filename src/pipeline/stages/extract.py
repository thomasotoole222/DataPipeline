"""The extract stage: fetch an endpoint's pages for a claimed task and land each one in ``raw.page``.

Public API:
    extract: Discard a task's landed pages, then fetch and land every page again, returning the records landed.
    FannedOut: What ``fan_out`` landed: the records and the distinct ids called.
    fan_out: Discard a task's landed pages, read its ids, then call the endpoint per id or batch and land every page.
"""

import asyncio
import logging
from collections import deque
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from datetime import timedelta
from itertools import islice
from types import MappingProxyType
from typing import Final, TypeGuard
from uuid import UUID

from pipeline.connectors.client import ConnectorClient
from pipeline.connectors.fanout import FromList, FromQuery, IdSource, id_text
from pipeline.connectors.fetch import Page, fetch_pages
from pipeline.connectors.spec import EndpointSpec
from pipeline.db import raw
from pipeline.db.control import ClaimedTask, Fence
from pipeline.db.lookup import read_column
from pipeline.db.pool import Pool, storable_text
from pipeline.db.raw import RawPage
from pipeline.errors import (
    RETRY_CAP,
    ClientErrorUnhandled,
    DatabaseConflict,
    DatabaseUnavailable,
    PermanentError,
    PipelineError,
    RateLimited,
    TooDeep,
    TransientError,
)
from pipeline.telemetry import Instruments
from pipeline.timeutil import TimeFrame, utc_now

#: Logger for this module; ``fan_out`` warns once per failed call.
_log: Final = logging.getLogger(__name__)
#: The longest ``retry_after`` the connector client waits out; a ``RateLimited`` asking for longer was not retried.
_LONG_WAIT: Final = timedelta(seconds=RETRY_CAP)

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
    requested, so a call they do not suit raises after the discard has committed. A ``RecursionError`` from the fetch
    or a page save, a response body nested deeper than the client can parse or clean, is raised as ``TooDeep``
    instead. Each page is stored with ``client.hashed_keys``.

    Preconditions:
        ``raw.page`` exists as ``raw.save_page`` requires. ``fence`` fences on ``task``'s lease. ``spec`` is an
        endpoint of ``task.connector``; ``spec.connector`` is stored with each page as given and is not compared with
        ``task.connector``.

    Args:
        pool: Pool each transaction takes a connection from.
        task: The claimed task; its ``id``, ``source_id`` and ``job`` are used.
        fence: Fences each transaction on the task's lease (the worker's ``Lease``).
        client: The client that sends each request; its ``hashed_keys`` say which part of each page is hashed.
        spec: The endpoint to fetch.
        window: The name of one of ``spec.windows``, or ``None`` when ``frame`` is ``None``.
        frame: The span of time to fetch, or ``None`` for an endpoint fetched without a window.
        instruments: Instruments whose ``rows`` counter is incremented per page.
        path_params: Values for the fields in ``spec.path``. Defaults to none.

    Returns:
        The sum of ``record_count`` over the pages landed, 0 or more.

    Raises:
        LeaseConflict: From ``fence.fenced`` when the lease is no longer held, before the discard or a page lands.
        TooDeep: When the fetch or a page save raises ``RecursionError``; the ``RecursionError`` is not chained.
        Exception: Whatever ``raw.discard_pages``, ``fetch_pages``, ``_land``, the pool, ``fence.fenced`` or the
            counter raises, unchanged. Every ``Exception`` escapes with ``rows_fetched`` set to the records landed
            before it. A ``BaseException`` that is not an ``Exception``, such as ``asyncio.CancelledError``, escapes
            without it.

    """
    rows = 0
    try:
        async with pool.connection() as conn, fence.fenced(conn):
            await raw.discard_pages(conn, task.id)  # every attempt starts over
        try:
            async for page in fetch_pages(client, spec, window, frame, task.source_id, path_params=path_params):
                await _land(pool, task, fence, spec, page, client.hashed_keys)
                rows += page.record_count
                instruments.rows.add(page.record_count, {"job": task.job, "stage": "fetched"})
        except RecursionError:  # its traceback is a thousand frames of one call, so it is not chained
            raise TooDeep(f"{spec.name}: a response body is nested too deeply to parse or store") from None
    except Exception as err:
        setattr(err, _ROWS_FETCHED, rows)
        raise
    return rows


async def _land(
    pool: Pool,
    task: ClaimedTask,
    fence: Fence,
    spec: EndpointSpec,
    page: Page,
    hashed_keys: frozenset[str],
    *,
    page_no: int | None = None,
    ids: tuple[str, ...] | None = None,
) -> None:
    """Save ``page`` as a ``RawPage`` row of ``task`` in ``raw.page``, in one transaction fenced on the task's lease.

    ``record_count`` is not stored. ``RawPage`` re-checks ``page_no`` and ``fetched_at``: ``Page`` already guarantees
    its own ``page_no`` is 0 or more and its ``fetched_at`` aware in UTC, and the ``page_no`` ``fan_out`` passes
    counts up from 0. A per-id page is stored under the ``page_no`` given and with the call's ``ids``; its cursors
    stay the call's own, so they resume nothing at the task's level.

    Preconditions:
        ``raw.page`` exists as ``raw.save_page`` requires.

    Args:
        pool: Pool the transaction takes a connection from.
        task: The task the page belongs to; its ``id`` is used.
        fence: Fences the transaction on the task's lease.
        spec: The spec that fetched the page; its ``connector`` and ``endpoint`` are stored with the page.
        page: The fetched page to save.
        hashed_keys: The body keys stored once per distinct content in ``raw.payload``, passed to ``raw.save_page``.
        page_no: The page's number in the task, or ``None`` for ``page.page_no``. Defaults to ``None``.
        ids: The ids of the fanned-out call the page belongs to, or ``None``. Defaults to ``None``.

    Returns:
        None.

    Raises:
        LeaseConflict: From ``fence.fenced`` when the lease is no longer held; the page is not saved.
        Exception: Whatever ``RawPage`` validation, the pool, ``fence.fenced`` or ``raw.save_page`` raises, unchanged.
            An existing row with the same ``task_id`` and ``page_no`` is overwritten, not rejected.

    """
    row = RawPage(
        task_id=task.id,
        page_no=page.page_no if page_no is None else page_no,
        connector=spec.connector,
        endpoint=spec.endpoint,
        cursor_in=page.cursor_in,
        cursor_next=page.cursor_next,
        fetched_at=page.fetched_at,
        payload=page.payload,
        ids=ids,
    )
    async with pool.connection() as conn, fence.fenced(conn):  # a connection only for the write, never across a call
        await raw.save_page(conn, row, hashed_keys)


@dataclass(frozen=True, slots=True, kw_only=True)
class FannedOut:
    """What ``fan_out`` landed. Frozen, slotted and keyword-only; no field is checked.

    Attributes:
        rows: The records landed: the list's records and every per-id call's, 0 or more.
        ids: The distinct ids the endpoint was called for, 0 when the id source gave none.

    """

    rows: int  # list records plus per-id records
    ids: int  # distinct ids called; 0: an empty id source, a normal day


@dataclass(frozen=True, slots=True)
class _Call:
    """One fanned-out call: the ids it was for, and its pages or why it failed. Frozen and slotted; nothing checked.

    Attributes:
        ids: The ids sent, in order.
        pages: The call's pages in order, numbered from 0 for the call alone; empty when it failed.
        error_class: The holdable error's ``error_class`` when the call failed, else ``None``.
        message: The holdable error's message when the call failed, else ``None``.

    """

    ids: tuple[str, ...]
    pages: tuple[Page, ...]
    error_class: str | None  # None: the call succeeded
    message: str | None


async def fan_out[C](
    pool: Pool,
    task: ClaimedTask,
    fence: Fence,
    client: ConnectorClient[C],
    spec: EndpointSpec,  # the per-id endpoint, with an id_param
    window: str | None,  # the job's window name, used on each spec that has windows
    frame: TimeFrame | None,  # None exactly when window is None
    instruments: Instruments,
    *,
    source: IdSource,
    concurrency: int,
) -> FannedOut:
    """Discard the task's landed pages, read its ids, then call ``spec`` once per id or batch and land every page.

    The discard runs as ``extract``'s does. A ``FromList`` source's list is then fetched with ``fetch_pages`` and
    each page landed as ``extract`` lands one, counted under ``stage="fetched"``, and the ids of the records of each
    page with a ``record_count`` above 0 read with ``FromList.id_of``. A ``FromQuery`` source's ids are read by
    ``_query_ids`` before anything is fetched. The distinct ids are sorted as text and split, in that order, into
    batches of ``spec.id_param.batch``; each batch is one call of ``spec`` by ``_calls``, at most ``concurrency`` in
    flight, each sent with ``fetch_pages(..., ids=batch)``. Each spec is fetched with ``window`` and ``frame`` when it
    has windows, and with neither when it has none. No connection is held across a connector call.

    Pages are landed in call order once a call has finished, each in its own fenced transaction, numbered on from the
    list's pages, so page numbers follow the ids' order whatever order the calls finish in; each per-id page is stored
    with the call's ``ids``, and its records are counted under ``stage="fanout_fetched"``. A call that fails with an
    error ``_holdable`` accepts lands one marker page instead, with the ids, the error's ``error_class`` and a payload
    of ``{"error": <message>}``; its ids are counted under ``stage="fanout_failed"`` and one warning names the error
    class and the number of ids. Any other error fails the task. A source with no ids makes no call. When an
    ``Exception`` escapes, the attribute named by ``_ROWS_FETCHED`` is set on it to the records landed so far, as in
    ``extract``, and a ``RecursionError`` is raised as ``TooDeep``.

    An id ``spec.request_for`` cannot place, such as a list record's id ``".."`` for a path ``id_param``, raises
    ``IdRefused`` before anything is sent; as a ``PermanentError`` it lands its call's marker like any holdable
    failure, so one bad id from outside data does not fail the task. A configuration error ``request_for`` or
    ``fetch_pages`` reports stays a ``ValueError`` and fails the task.

    Preconditions:
        As ``extract``. ``spec`` has an ``id_param``; a ``FromList`` source's ``spec`` is of the same connector.
        ``concurrency`` is 1 or more. ``BatchJob.__init__`` checks all three.

    Args:
        pool: Pool each transaction, and the id query, takes a connection from.
        task: The claimed task; its ``id``, ``source_id``, ``job`` and ``partition_key`` are used.
        fence: Fences each write on the task's lease.
        client: The client that sends each request; its ``hashed_keys`` say which part of each page is hashed.
        spec: The endpoint called per id or batch.
        window: The window name of the job, or ``None`` when ``frame`` is ``None``.
        frame: The task's time frame, or ``None``; also bound into a ``FromQuery`` statement.
        instruments: Instruments whose ``rows`` counter is incremented per page and per failed call.
        source: Where the ids come from.
        concurrency: The most calls in flight at once.

    Returns:
        A ``FannedOut`` with the records landed and the distinct ids called.

    Raises:
        LeaseConflict: From ``fence.fenced`` when the lease is no longer held, before the discard or a page lands.
        TooDeep: When a fetch or a page save raises ``RecursionError``; the ``RecursionError`` is not chained.
        ClientErrorUnhandled: From ``FromList.id_of``, for a list record with no readable id; or from the list's
            fetch.
        ValueError: From ``_query_ids``, for a query that does not return one column of ids.
        Exception: Whatever the discard, the list's fetch, ``_query_ids``, a call's non-holdable error (such as
            ``AuthBlocked``, ``LeaseConflict``, a ``RateLimited`` asking to wait longer than ``RETRY_CAP``, a
            database error or a ``ValueError``), ``_land``, the pool, ``fence.fenced`` or the counter raises,
            unchanged. Every ``Exception`` escapes with ``rows_fetched`` set to the records landed before it; calls
            still in flight are cancelled first.

    """
    rows = 0
    try:
        async with pool.connection() as conn, fence.fenced(conn):
            await raw.discard_pages(conn, task.id)  # every attempt starts over
        try:
            page_no = 0  # the next page's number across the task
            ids: set[str] = set()
            if isinstance(source, FromList):
                list_window, list_frame = _windowed(source.spec, window, frame)
                async for page in fetch_pages(client, source.spec, list_window, list_frame, task.source_id):
                    await _land(pool, task, fence, source.spec, page, client.hashed_keys)
                    rows += page.record_count
                    instruments.rows.add(page.record_count, {"job": task.job, "stage": "fetched"})
                    if page.record_count:  # a last page may leave out its empty record list
                        ids.update(source.id_of(record) for record in source.spec.records(page.payload))
                    page_no = page.page_no + 1
            else:
                ids = await _query_ids(pool, task, frame, source)
            ordered = sorted(ids)
            batch = 1 if spec.id_param is None else spec.id_param.batch
            batches = [tuple(ordered[at : at + batch]) for at in range(0, len(ordered), batch)]
            call_window, call_frame = _windowed(spec, window, frame)
            calls = _calls(client, spec, call_window, call_frame, task.source_id, batches, concurrency)
            async with aclosing(calls) as landing:
                async for call in landing:
                    if call.error_class is not None:
                        await _land_failure(pool, task, fence, spec, call, page_no, client.hashed_keys)
                        instruments.rows.add(len(call.ids), {"job": task.job, "stage": "fanout_failed"})
                        page_no += 1
                        continue
                    for page in call.pages:
                        await _land(pool, task, fence, spec, page, client.hashed_keys, page_no=page_no, ids=call.ids)
                        rows += page.record_count
                        instruments.rows.add(page.record_count, {"job": task.job, "stage": "fanout_fetched"})
                        page_no += 1
        except RecursionError:  # its traceback is a thousand frames of one call, so it is not chained
            raise TooDeep(f"{spec.name}: a response body is nested too deeply to parse or store") from None
    except Exception as err:
        setattr(err, _ROWS_FETCHED, rows)
        raise
    return FannedOut(rows=rows, ids=len(ordered))


def _windowed(spec: EndpointSpec, window: str | None, frame: TimeFrame | None) -> tuple[str | None, TimeFrame | None]:
    """Return the window name and frame a spec of a fanned-out job is fetched with.

    Preconditions:
        None.

    Args:
        spec: The list or per-id endpoint.
        window: The job's window name, or ``None``.
        frame: The task's time frame, or ``None``.

    Returns:
        ``(window, frame)`` when ``spec`` has windows; ``(None, None)`` when it has none, so it is fetched without the
        frame.

    Raises:
        Nothing.

    """
    return (window, frame) if spec.windows else (None, None)


async def _query_ids(pool: Pool, task: ClaimedTask, frame: TimeFrame | None, source: FromQuery) -> set[str]:
    """Run a ``FromQuery`` statement read-only on one pooled connection and return its distinct ids as text.

    The named parameters bound are those ``QUERY_PARAMS`` lists: the task's ``source_id`` and ``partition_key``, and
    ``frame_start`` and ``frame_end``, the frame's bounds or ``None`` without a frame.

    Preconditions:
        None.

    Args:
        pool: Pool the connection is taken from.
        task: The claimed task; its ``source_id`` and ``partition_key`` are bound.
        frame: The task's time frame, or ``None``.
        source: The query.

    Returns:
        The distinct ids, each written by ``id_text``; empty when the query returns no row.

    Raises:
        ValueError: From ``read_column`` when the query does not return one column; or when a value is not a
            non-empty text or an integer (``NULL`` included), naming the job but not the value.
        DatabaseConflict: From ``read_column``.
        DatabaseUnavailable: From ``read_column``.
        psycopg.Error: From ``read_column``, any other database error unchanged.

    """
    params = {
        "source_id": task.source_id,
        "partition_key": task.partition_key,
        "frame_start": None if frame is None else frame.start,
        "frame_end": None if frame is None else frame.end,
    }
    async with pool.connection() as conn:
        values = await read_column(conn, source.sql, params)
    ids: set[str] = set()
    for value in values:
        text = id_text(value)
        if text is None:
            kind = type(value).__name__
            raise ValueError(f"job {task.job}: the id query returned a {kind}, not a text or integer id")
        ids.add(text)
    return ids


async def _calls[C](
    client: ConnectorClient[C],
    spec: EndpointSpec,
    window: str | None,
    frame: TimeFrame | None,
    source_id: UUID,
    batches: Sequence[tuple[str, ...]],
    concurrency: int,
) -> AsyncGenerator[_Call]:
    """Run one ``_call`` per batch, at most ``concurrency`` at once, and yield them in batch order as each finishes.

    The first ``concurrency`` calls start at once; only when the oldest call has finished does the next batch's call
    start, so a call that finishes early waits, in memory, for those before it, and holds its slot until then. While
    the caller lands a yielded call, up to ``concurrency`` more are running, so at most ``concurrency + 1`` calls'
    pages are in memory. When the generator stops early (a call's error, an error in the caller, cancellation, or
    ``aclose``), every call still running is cancelled and awaited, its result or error discarded, so none outlives it.

    A slow oldest call (one the client is retrying with backoff) therefore keeps the other slots idle once their calls
    finish, and the fan-out runs one call at a time behind it (todo.txt 18).

    Preconditions:
        ``concurrency`` is 1 or more. The caller closes the generator (``aclosing``) when it stops iterating early.

    Args:
        client: The client that sends each request.
        spec: The per-id endpoint.
        window: The window name it is fetched with, or ``None``.
        frame: The frame it is fetched with, or ``None``.
        source_id: The source the requests are made for.
        batches: The ids of each call, in call order.
        concurrency: The most calls in flight at once.

    Yields:
        Each call's ``_Call``, in batch order.

    Raises:
        Exception: Whatever a call raises that ``_call`` does not hold, unchanged, once the calls before it have been
            yielded; the calls still running are cancelled first.
        asyncio.CancelledError: When the task iterating it is cancelled while it awaits a call; that call and the
            others running are cancelled too.

    """
    upcoming = iter(batches)
    running: deque[asyncio.Task[_Call]] = deque()
    try:
        for batch in islice(upcoming, concurrency):
            running.append(asyncio.create_task(_call(client, spec, window, frame, source_id, batch)))
        while running:
            call = await running[0]
            running.popleft()
            for batch in islice(upcoming, 1):  # a slot is free: start the next call while this one lands
                running.append(asyncio.create_task(_call(client, spec, window, frame, source_id, batch)))
            yield call
    finally:
        for pending in running:
            pending.cancel()
        await asyncio.gather(*running, return_exceptions=True)


async def _call[C](
    client: ConnectorClient[C],
    spec: EndpointSpec,
    window: str | None,
    frame: TimeFrame | None,
    source_id: UUID,
    ids: tuple[str, ...],
) -> _Call:
    """Fetch every page of one call of ``spec`` for some ids, or describe the holdable error it failed with.

    The pages are kept in memory until the call is done, so a call that fails part-way lands none of them.

    Preconditions:
        ``spec`` has an ``id_param`` and ``ids`` holds 1 to ``id_param.batch`` ids.

    Args:
        client: The client that sends each request.
        spec: The per-id endpoint.
        window: The window name it is fetched with, or ``None``.
        frame: The frame it is fetched with, or ``None``.
        source_id: The source the requests are made for.
        ids: The ids the call is for.

    Returns:
        A ``_Call`` with every page, or with no pages and the error's ``error_class`` and message when ``fetch_pages``
        raised an error ``_holdable`` accepts.

    Raises:
        Exception: Whatever ``fetch_pages`` raises that ``_holdable`` refuses, unchanged.

    """
    try:
        pages = tuple([page async for page in fetch_pages(client, spec, window, frame, source_id, ids=ids)])
    except Exception as err:
        if not _holdable(err):
            raise
        return _Call(ids, (), err.error_class, str(err))
    return _Call(ids, pages, None, None)


def _holdable(err: Exception) -> TypeGuard[PipelineError]:
    """Return whether a call's error is held as ``detail_failed`` for its ids instead of failing the task.

    Held: a ``PermanentError`` (``RowLimitReached`` and ``IdRefused`` included), a ``ClientErrorUnhandled``, and a
    ``TransientError`` other than those below, which ``HttpConnectorClient.send`` has already retried (a
    ``RateLimited`` without a ``retry_after``, or with one of at most ``RETRY_CAP`` seconds, included). Not held: a
    ``RateLimited`` whose ``retry_after`` is longer than ``RETRY_CAP``, which the client never retried and every other
    call would meet too; a ``DatabaseUnavailable`` or ``DatabaseConflict``, which is the warehouse's failure, not the
    call's; and anything else, such as ``AuthBlocked``, ``LeaseConflict``, ``ValueError`` (a configuration error) or
    ``RecursionError``.

    Preconditions:
        None.

    Args:
        err: The error a call raised.

    Returns:
        Whether it is held.

    Raises:
        Nothing.

    """
    if isinstance(err, RateLimited) and err.retry_after is not None and err.retry_after > _LONG_WAIT:
        return False
    if isinstance(err, DatabaseUnavailable | DatabaseConflict):
        return False
    return isinstance(err, PermanentError | ClientErrorUnhandled | TransientError)


async def _land_failure(
    pool: Pool,
    task: ClaimedTask,
    fence: Fence,
    spec: EndpointSpec,
    call: _Call,
    page_no: int,
    hashed_keys: frozenset[str],
) -> None:
    """Land a failed call's marker page and log one warning naming its error class and the number of its ids.

    The warning is logged before the marker is saved. The marker has no cursors, ``fetched_at`` set to when it is
    landed (not when the call failed), a payload of ``{"error": <message>}``, the call's ``ids`` and its
    ``error_class``. The message is passed through ``storable_text``, so a NUL or lone surrogate in it (such as from
    the first 500 characters of a non-JSON error body a ``ClientErrorUnhandled`` quotes) is replaced with U+FFFD and
    the marker still lands, instead of ``jsonb`` refusing it and failing the task on every attempt.

    Preconditions:
        ``call.error_class`` and ``call.message`` are set. ``raw.page`` exists as ``raw.save_page`` requires.

    Args:
        pool: Pool the transaction takes a connection from.
        task: The task the page belongs to; its ``id`` and ``job`` are used.
        fence: Fences the transaction on the task's lease.
        spec: The per-id endpoint; its ``connector`` and ``endpoint`` are stored with the page.
        call: The failed call.
        page_no: The marker's page number in the task.
        hashed_keys: Passed to ``raw.save_page``.

    Returns:
        None.

    Raises:
        LeaseConflict: From ``fence.fenced`` when the lease is no longer held; the marker is not saved.
        Exception: Whatever ``RawPage`` validation, the pool, ``fence.fenced`` or ``raw.save_page`` raises, unchanged,
            such as a psycopg ``DataError`` for a message ``jsonb`` rejects. An existing row with the same ``task_id``
            and ``page_no`` is overwritten.

    """
    _log.warning("task %s %s: a call for %d ids failed with %s", task.id, task.job, len(call.ids), call.error_class)
    row = RawPage(
        task_id=task.id,
        page_no=page_no,
        connector=spec.connector,
        endpoint=spec.endpoint,
        cursor_in=None,
        cursor_next=None,
        fetched_at=utc_now(),
        payload={"error": storable_text(call.message or "")},
        ids=call.ids,
        error_class=call.error_class,
    )
    async with pool.connection() as conn, fence.fenced(conn):
        await raw.save_page(conn, row, hashed_keys)
