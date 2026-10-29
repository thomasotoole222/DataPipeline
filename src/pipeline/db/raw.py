"""The ``raw`` schema: page validation, and saving, reading and discarding fetched pages in ``raw.page``.

A page is stored in two parts: its hashed part (the body's hashed keys) once per distinct content in ``raw.payload``,
and the rest of the body, its envelope, in ``raw.page``.

Public API:
    RawPage: One page of results fetched for a task, as stored in ``raw.page``.
    save_page: Store a ``RawPage``'s hashed part in ``raw.payload`` and the page in ``raw.page``, overwriting the page.
    last_page: Return the highest-numbered page stored for a task, or ``None``.
    read_pages: Fetch every page of one task in one query, then yield them as ``RawPage`` objects by ``page_no``.
    discard_pages: Delete every row of a task from ``raw.page`` and return the cursor's row count.

"""

import hashlib
import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

from psycopg.rows import class_row
from psycopg.types.json import Jsonb

from pipeline.db.pool import Connection
from pipeline.errors import db_errors
from pipeline.timeutil import ensure_utc


@dataclass(frozen=True, slots=True, kw_only=True)
class RawPage:
    """One page of results fetched for a task, as stored in ``raw.page``.

    Instances are frozen, use slots, and take every field as a required keyword argument in the generated
    ``__init__``. ``__post_init__`` checks that ``page_no`` is 0 or more and that ``payload`` is a ``dict``, and
    replaces ``fetched_at`` with ``ensure_utc(fetched_at)``, so every instance holds an aware UTC ``fetched_at``.
    ``last_page`` and ``read_pages`` build instances from ``raw.page`` rows, so those rows pass the same checks.
    The generated ``__eq__`` compares all fields. The generated ``__hash__`` hashes all fields, so it always raises
    ``TypeError`` because ``payload`` is a ``dict``: instances cannot go in a set or be dict keys.

    Attributes:
        task_id (int): Identifier of the ``control.task`` the page belongs to. Not validated.
        page_no (int): Position of the page within the task: 0 for the first page, then 1, 2 and so on. At least 0.
        connector (str): The connector that fetched the page, e.g. ``"tiktok_shop"``. Not validated.
        endpoint (str): The endpoint within the connector, e.g. ``"orders_search"``. Not validated.
        cursor_in (str | None): Cursor sent to fetch this page, or ``None``. Not validated.
        cursor_next (str | None): Cursor returned for the next page, or ``None``. Not validated.
        fetched_at (datetime): Time the response arrived, aware and in UTC.
        payload (dict[str, Any]): The whole decoded response body, envelope included. Must be a ``dict``; its
            contents are not checked, and the dict passed in is stored as is, not copied.

    """

    task_id: int  # control.task id of the task that fetched the page
    page_no: int  # 0 for the first page, then 1, 2, ...
    connector: str  # the connector that fetched it, e.g. "tiktok_shop"
    endpoint: str  # the endpoint within the connector, e.g. "orders_search"
    cursor_in: str | None  # cursor sent; None on the first page
    cursor_next: str | None  # cursor for the next page; None on the last page
    fetched_at: datetime  # when the response arrived, converted to UTC
    payload: dict[str, Any]  # the whole response body, envelope included

    def __post_init__(self) -> None:
        """Validate ``page_no`` and ``payload`` and normalise ``fetched_at`` to UTC.

        Checks run in order: ``page_no``, then ``payload``, then ``fetched_at`` is replaced with
        ``ensure_utc(fetched_at)``.

        Preconditions:
            The fields hold their annotated types (mypy checks callers; rows from ``raw.page`` match by its column
            types and ``NOT NULL`` constraints).

        Args:
            None.

        Returns:
            None.

        Raises:
            ValueError: If ``page_no`` is less than ``0``, or, from ``ensure_utc``, if ``fetched_at`` is naive.
            TypeError: If ``payload`` is not a ``dict``, or, from ``ensure_utc``, if ``fetched_at`` is not a
                ``datetime``.
            OverflowError: From ``ensure_utc``, if converting ``fetched_at`` to UTC leaves the ``datetime`` range.

        """
        if self.page_no < 0:
            raise ValueError(f"page_no must be 0 or more, got {self.page_no}")
        if not isinstance(self.payload, dict):
            raise TypeError(f"payload must be a JSON object, got {type(self.payload).__name__}")
        object.__setattr__(self, "fetched_at", ensure_utc(self.fetched_at))


#: Insert statement for one ``raw.payload`` row; the two ``%s`` parameters are ``hash`` and ``body``. A row with the
#: same hash already holds the same body, so a conflict leaves it as it is.
_SAVE_PAYLOAD: Final = "INSERT INTO raw.payload (hash, body) VALUES (%s, %s) ON CONFLICT (hash) DO NOTHING"

#: Upsert statement for one ``raw.page`` row. The nine ``%s`` parameters are, in order, ``task_id``, ``page_no``,
#: ``connector``, ``endpoint``, ``cursor_in``, ``cursor_next``, ``fetched_at``, ``envelope`` and ``payload_hash``. On a
#: conflict on ``(task_id, page_no)`` it overwrites every other column with the incoming values.
_SAVE: Final = (
    "INSERT INTO raw.page "
    "(task_id, page_no, connector, endpoint, cursor_in, cursor_next, fetched_at, envelope, payload_hash) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
    "ON CONFLICT (task_id, page_no) DO UPDATE SET connector = EXCLUDED.connector, endpoint = EXCLUDED.endpoint, "
    "cursor_in = EXCLUDED.cursor_in, cursor_next = EXCLUDED.cursor_next, fetched_at = EXCLUDED.fetched_at, "
    "envelope = EXCLUDED.envelope, payload_hash = EXCLUDED.payload_hash"
)


def _canonical(value: Any) -> bytes:
    """Return the canonical JSON of a value: the bytes ``raw.payload``'s ``hash`` is the sha256 of.

    The form is ``json.dumps`` with keys sorted, no whitespace (``","`` and ``":"`` separators), every character
    outside ASCII escaped and ``NaN`` and infinities refused, encoded as ASCII. Two values that are equal as JSON give
    the same bytes, whatever the order of their keys.

    Preconditions:
        None.

    Args:
        value: A JSON value: dicts with ``str`` keys, lists, strings, numbers, booleans and ``None``.

    Returns:
        The canonical JSON as ASCII bytes.

    Raises:
        TypeError: From ``json.dumps``, when ``value`` holds a value JSON cannot represent or a dict key that is not a
            ``str``, ``int``, ``float``, ``bool`` or ``None``.
        ValueError: From ``json.dumps``, when ``value`` holds a ``NaN`` or infinite float, or contains itself.

    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _split(body: Mapping[str, Any], hashed_keys: frozenset[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a response body into its envelope and its hashed part by top-level key.

    Preconditions:
        None.

    Args:
        body: The decoded response body.
        hashed_keys: The connector's hashed keys; a key ``body`` does not have is left out of both parts.

    Returns:
        ``(envelope, hashed)``: the body's items whose key is not in ``hashed_keys``, and those whose key is, each a
        new ``dict`` holding the body's own values. Together they hold every item of ``body`` once.

    Raises:
        Nothing.

    """
    envelope = {key: value for key, value in body.items() if key not in hashed_keys}
    hashed = {key: value for key, value in body.items() if key in hashed_keys}
    return envelope, hashed


async def save_page(conn: Connection, page: RawPage, hashed_keys: frozenset[str]) -> None:
    """Store one raw page: its hashed part in ``raw.payload`` once, then the page in ``raw.page``.

    The payload is split by ``_split``: the items whose key is in ``hashed_keys`` are the hashed part, the rest the
    envelope. The hashed part is inserted into ``raw.payload`` under the sha256 of its ``_canonical`` JSON with
    ``_SAVE_PAYLOAD``, which leaves an existing row with that hash as it is; then the page row is upserted with
    ``_SAVE``, its envelope and that hash. An existing row with the same ``task_id`` and ``page_no`` has every other
    column overwritten. Both statements run on ``conn`` inside ``db_errors()``; the function does not commit, so the
    two are atomic only inside the caller's transaction. Both JSON parts are wrapped in ``Jsonb``, which serialises
    them with ``json.dumps``. ``RawPage`` guarantees ``page_no`` meets the table's ``CHECK (page_no >= 0)`` and
    ``fetched_at`` is aware.

    ``jsonb`` rejects a string holding NUL (U+0000) or a lone surrogate, and ``_canonical`` and ``jsonb`` reject a
    ``NaN``/``Infinity`` number. A page fetched through ``HttpConnectorClient`` holds none of them: the client
    replaces NUL and lone surrogates with U+FFFD and refuses a 2xx body holding a non-standard constant, so only a
    payload built some other way can fail here.

    Preconditions:
        ``conn`` is an open asynchronous connection that is not running another statement. ``raw.payload`` and
        ``raw.page`` exist as ``pipeline_raw_quarantine.sql`` creates them. ``page.payload`` is JSON-serialisable by
        ``json.dumps`` and holds no value ``jsonb`` rejects. The caller commits afterwards unless ``conn`` is in
        autocommit mode.

    Args:
        conn: Connection the statements run on.
        page: Page whose ``task_id``, ``page_no``, ``connector``, ``endpoint``, ``cursor_in``, ``cursor_next``,
            ``fetched_at`` and ``payload`` are written.
        hashed_keys: The top-level keys of the payload stored in ``raw.payload``: the connector's ``hashed_keys``.

    Returns:
        None.

    Raises:
        DatabaseConflict: From ``db_errors``, on a serialisation failure or deadlock.
        DatabaseUnavailable: From ``db_errors``, when the connection is lost or the database cannot be reached.
        TypeError: From ``_canonical``, when the hashed part holds a value JSON cannot represent.
        ValueError: From ``_canonical``, when the hashed part holds a ``NaN`` or infinite float.
        Exception: Any other error from ``conn.execute`` propagates unchanged, for example a missing table, a
            ``TypeError`` from serialising the envelope, a psycopg ``DataError`` from ``jsonb`` input, or a
            ``ForeignKeyViolation`` when ``control.purge_raw`` deleted the payload row between the two statements.

    """
    envelope, hashed = _split(page.payload, hashed_keys)
    digest = hashlib.sha256(_canonical(hashed)).digest()
    params = (
        page.task_id,
        page.page_no,
        page.connector,
        page.endpoint,
        page.cursor_in,
        page.cursor_next,
        page.fetched_at,
        Jsonb(envelope),
        digest,
    )
    with db_errors():
        await conn.execute(_SAVE_PAYLOAD, (digest, Jsonb(hashed)))
        await conn.execute(_SAVE, params)


#: Query that selects the ``raw.page`` row with the highest ``page_no`` for one ``task_id``, its envelope merged with
#: its ``raw.payload`` body back into the whole payload. ``%s`` is the ``task_id`` parameter, bound at execution.
_LAST: Final = (
    "SELECT p.task_id, p.page_no, p.connector, p.endpoint, p.cursor_in, p.cursor_next, p.fetched_at, "
    "p.envelope || y.body AS payload "
    "FROM raw.page AS p JOIN raw.payload AS y ON y.hash = p.payload_hash "
    "WHERE p.task_id = %s ORDER BY p.page_no DESC LIMIT 1"
)


async def last_page(conn: Connection, task_id: int) -> RawPage | None:
    """Return the highest-numbered page stored for a task.

    Opens a cursor on ``conn`` whose row factory is ``class_row(RawPage)``, executes ``_LAST`` with ``task_id``
    bound, and fetches one row. Opening, executing and fetching run inside ``db_errors()``, which never swallows an
    exception.

    Preconditions:
        ``conn`` is an open asynchronous connection that no other coroutine is using concurrently. ``raw.page`` and
        ``raw.payload`` exist. ``_LAST`` selects exactly the fields ``RawPage`` takes as keyword arguments.

    Args:
        conn: Connection to run the query on.
        task_id: Task whose last page is fetched.

    Returns:
        The ``RawPage`` built from the row with the highest ``page_no`` for ``task_id``, its ``payload`` the page's
        envelope merged with its ``raw.payload`` body, or ``None`` if ``raw.page`` has no row for ``task_id``.

    Raises:
        DatabaseConflict: From ``db_errors``, on a serialisation failure or deadlock.
        DatabaseUnavailable: From ``db_errors``, when the connection is lost or the database cannot be reached.
        Exception: Any other error from the cursor, or from ``RawPage`` building the row, propagates unchanged.

    """
    with db_errors():
        async with conn.cursor(row_factory=class_row(RawPage)) as cursor:
            await cursor.execute(_LAST, (task_id,))
            return await cursor.fetchone()


#: Query that selects one task's ``raw.page`` rows, ordered by ``page_no`` ascending, each envelope merged with its
#: ``raw.payload`` body back into the whole payload. The ``%s`` placeholder takes the task id.
_READ: Final = (
    "SELECT p.task_id, p.page_no, p.connector, p.endpoint, p.cursor_in, p.cursor_next, p.fetched_at, "
    "p.envelope || y.body AS payload "
    "FROM raw.page AS p JOIN raw.payload AS y ON y.hash = p.payload_hash "
    "WHERE p.task_id = %s ORDER BY p.page_no"
)


async def read_pages(conn: Connection, task_id: int) -> AsyncIterator[RawPage]:
    """Yield the pages of one task from ``raw.page`` in ``page_no`` order.

    Nothing runs until iteration starts. On the first step, every matching row is fetched with one cursor on
    ``conn`` inside ``db_errors()``, built into a ``RawPage`` by ``class_row``, and the cursor is closed; the pages
    are then yielded from memory, so the whole task is held in memory at once.

    Preconditions:
        ``conn`` is open and not used concurrently until the first page has been yielded. ``raw.page`` and
        ``raw.payload`` exist.
        ``_READ`` selects exactly the fields ``RawPage`` takes as keyword arguments.

    Args:
        conn: Connection the cursor is opened on.
        task_id: Task whose rows are selected.

    Yields:
        One ``RawPage`` per row whose ``task_id`` matches, in ascending ``page_no`` order, its ``payload`` the page's
        envelope merged with its ``raw.payload`` body: the body ``save_page`` was given, as ``jsonb`` stores it.
        Nothing if no row matches, so an unknown task is not distinguished from a task with no pages.

    Raises:
        DatabaseConflict: From ``db_errors``, on a serialisation failure or deadlock.
        DatabaseUnavailable: From ``db_errors``, when the connection is lost or the database cannot be reached.
        Exception: Any other error from the cursor, or from ``RawPage`` building a row, propagates unchanged. Every
            exception surfaces on the first iteration step, never at the call itself.

    """
    with db_errors():
        async with conn.cursor(row_factory=class_row(RawPage)) as cursor:
            await cursor.execute(_READ, (task_id,))
            pages = await cursor.fetchall()
    for page in pages:
        yield page


#: Statement that deletes every ``raw.page`` row matching one ``task_id``. ``%s`` is bound to the task id when the
#: statement is executed.
_DISCARD: Final = "DELETE FROM raw.page WHERE task_id = %s"


async def discard_pages(conn: Connection, task_id: int) -> int:
    """Delete every row in ``raw.page`` whose ``task_id`` column equals ``task_id``.

    Executes ``_DISCARD`` on ``conn`` inside ``db_errors()`` and returns the cursor's ``rowcount``. ``db_errors``
    never swallows an exception, so ``cursor`` is always bound when it is read. The function does not commit and does
    not close the cursor. The pages' ``raw.payload`` rows are kept; ``control.purge_raw`` deletes those no page
    references.

    Preconditions:
        ``conn`` is an open asynchronous connection that is not running another statement. ``raw.page`` exists. If
        ``conn`` is not in autocommit mode, the caller commits or rolls back afterwards.

    Args:
        conn: Connection the ``DELETE`` runs on.
        task_id: Value matched against the ``task_id`` column.

    Returns:
        The number of rows the ``DELETE`` removed, ``0`` when no row matched.

    Raises:
        DatabaseConflict: From ``db_errors``, on a serialisation failure or deadlock.
        DatabaseUnavailable: From ``db_errors``, when the connection is lost or the database cannot be reached.
        Exception: Any other error from ``conn.execute`` propagates unchanged.

    """
    with db_errors():
        cursor = await conn.execute(_DISCARD, (task_id,))
    return cursor.rowcount
