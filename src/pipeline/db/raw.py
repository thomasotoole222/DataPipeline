"""The ``raw`` schema: page validation, and saving, reading and discarding fetched pages in ``raw.page``.

Public API:
    RawPage: One page of results fetched for a task, as stored in ``raw.page``.
    save_page: Insert a ``RawPage`` row, or overwrite the existing row with the same ``task_id`` and ``page_no``.
    last_page: Return the highest-numbered page stored for a task, or ``None``.
    read_pages: Stream the pages of one task as ``RawPage`` objects, ordered by ``page_no``.
    discard_pages: Delete every row of a task from ``raw.page`` and return the reported row count.
"""

from collections.abc import AsyncIterator
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
    ``__init__``. ``__post_init__`` validates ``page_no`` and ``payload`` and replaces ``fetched_at`` with the result
    of ``ensure_utc``. The generated ``__eq__`` compares all fields and the generated ``__hash__`` hashes all fields.
    A dataclass subclass must also be frozen.

    Attributes:
        task_id (int): Identifier of the task the page belongs to. Not validated.
        page_no (int): Position of the page within the task, 0 or more.
        connector (str): The connector that fetched the page, e.g. ``"tiktok_shop"``. Not validated.
        endpoint (str): The endpoint within the connector, e.g. ``"orders_search"``. Not validated.
        cursor_in (str | None): Cursor sent to fetch this page, or ``None``. Not validated.
        cursor_next (str | None): Cursor returned for the next page, or ``None``. Not validated.
        fetched_at (datetime): Time the page was fetched, as returned by ``ensure_utc``.
        payload (dict[str, Any]): Decoded page body. The dict passed in is stored as is, not copied.

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
            ``page_no`` supports ``<`` against ``0``. ``fetched_at`` is a value ``ensure_utc`` accepts.

        Args:
            None.

        Returns:
            None.

        Raises:
            ValueError: If ``page_no`` is less than ``0``.
            TypeError: If ``payload`` is not a ``dict``. Also let through, with an unrelated message, when ``page_no``
                cannot be compared with ``0`` (for example a ``str`` or ``None``).
            Exception: Whatever ``ensure_utc`` raises is let through.

        """
        if self.page_no < 0:
            raise ValueError(f"page_no must be 0 or more, got {self.page_no}")
        if not isinstance(self.payload, dict):
            raise TypeError(f"payload must be a JSON object, got {type(self.payload).__name__}")
        object.__setattr__(self, "fetched_at", ensure_utc(self.fetched_at))


#: Upsert statement for one ``raw.page`` row. The eight ``%s`` parameters are, in order, ``task_id``, ``page_no``,
#: ``connector``, ``endpoint``, ``cursor_in``, ``cursor_next``, ``fetched_at`` and ``payload``. On a conflict on
#: ``(task_id, page_no)`` it overwrites every other column with the incoming values.
_SAVE: Final = (
    "INSERT INTO raw.page (task_id, page_no, connector, endpoint, cursor_in, cursor_next, fetched_at, payload) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
    "ON CONFLICT (task_id, page_no) DO UPDATE SET connector = EXCLUDED.connector, endpoint = EXCLUDED.endpoint, "
    "cursor_in = EXCLUDED.cursor_in, "
    "cursor_next = EXCLUDED.cursor_next, fetched_at = EXCLUDED.fetched_at, payload = EXCLUDED.payload"
)


async def save_page(conn: Connection, page: RawPage) -> None:
    """Upsert one raw page into ``raw.page``.

    Executes ``_SAVE`` on ``conn`` inside ``db_errors()``. An existing row with the same ``task_id`` and ``page_no``
    has its ``connector``, ``endpoint``, ``cursor_in``, ``cursor_next``, ``fetched_at`` and ``payload`` overwritten.
    ``payload`` is wrapped in ``Jsonb``. The function does not commit.

    Preconditions:
        ``conn`` is an open asynchronous connection that is not running another statement. ``raw.page`` exists as
        ``pipeline_raw_quarantine.sql`` creates it. ``page.payload`` is JSON-serialisable. The caller commits
        afterwards unless ``conn`` is in autocommit mode.

    Args:
        conn: Connection the statement runs on.
        page: Page whose ``task_id``, ``page_no``, ``connector``, ``endpoint``, ``cursor_in``, ``cursor_next``,
            ``fetched_at`` and ``payload`` are written.

    Returns:
        None.

    Raises:
        Whatever ``db_errors`` raises or lets through when ``conn.execute`` fails, for example on a missing table or
        a payload that cannot be serialised; conditional on ``db_errors``.

    """
    params = (
        page.task_id,
        page.page_no,
        page.connector,
        page.endpoint,
        page.cursor_in,
        page.cursor_next,
        page.fetched_at,
        Jsonb(page.payload),
    )
    with db_errors():
        await conn.execute(_SAVE, params)


#: Query that selects the ``raw.page`` row with the highest ``page_no`` for one ``task_id``. ``%s`` is the
#: ``task_id`` parameter, bound at execution.
_LAST: Final = (
    "SELECT task_id, page_no, connector, endpoint, cursor_in, cursor_next, fetched_at, payload "
    "FROM raw.page WHERE task_id = %s ORDER BY page_no DESC LIMIT 1"
)


async def last_page(conn: Connection, task_id: int) -> RawPage | None:
    """Return the highest-numbered page row stored for a task.

    Opens a cursor on ``conn`` whose row factory is ``class_row(RawPage)``, executes ``_LAST`` with ``task_id``
    bound, and fetches one row. Execution and fetching run inside ``db_errors()``.

    Preconditions:
        ``conn`` is an open asynchronous connection that no other coroutine is using concurrently. ``_LAST`` selects
        exactly the fields that ``RawPage`` accepts as keyword arguments.

    Args:
        conn: Connection to run the query on.
        task_id: Task whose last page is fetched.

    Returns:
        The ``RawPage`` built from the row with the highest ``page_no`` for ``task_id``. ``None`` if ``raw.page`` has
        no row for ``task_id``, and also ``None`` if ``db_errors`` suppresses an exception.

    Raises:
        Exceptions raised while opening the cursor, executing or fetching, as translated or re-raised by
        ``db_errors``. This function swallows nothing itself. Whether ``db_errors`` swallows anything depends on its
        implementation.

    """
    with db_errors():
        async with conn.cursor(row_factory=class_row(RawPage)) as cursor:
            await cursor.execute(_LAST, (task_id,))
            return await cursor.fetchone()


#: Query that selects one task's ``raw.page`` rows, ordered by ``page_no`` ascending. The ``%s`` placeholder takes
#: the task id.
_READ: Final = (
    "SELECT task_id, page_no, connector, endpoint, cursor_in, cursor_next, fetched_at, payload "
    "FROM raw.page WHERE task_id = %s ORDER BY page_no"
)


async def read_pages(conn: Connection, task_id: int) -> AsyncIterator[RawPage]:
    """Read the pages of one task from ``raw.page``.

    Nothing runs until iteration starts. On the first step, all matching rows are fetched with a single cursor on
    ``conn`` inside ``db_errors()``, converted to ``RawPage`` by ``class_row``, and the cursor is closed; the pages are
    then yielded from memory.

    Preconditions:
        ``conn`` is open and not used concurrently until the first page has been yielded.
        ``RawPage`` accepts the columns selected by ``_READ`` as keyword arguments.

    Args:
        conn: Connection the cursor is opened on.
        task_id: Task whose rows are selected.

    Yields:
        One ``RawPage`` per row whose ``task_id`` matches, in ascending ``page_no`` order. Yields nothing if no row
        matches, so an unknown task is not distinguished from a task with no pages.

    Raises:
        Whatever ``db_errors`` translates database errors into, from executing the query or fetching rows. Errors
        from ``RawPage`` that ``db_errors`` does not handle propagate unchanged. All exceptions surface on the first
        iteration step, never at the call itself.

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

    Executes ``_DISCARD`` on ``conn`` inside ``db_errors()`` and returns the cursor's ``rowcount``. The function does
    not commit and does not close the cursor.

    Preconditions:
        ``conn`` is open and accepts an awaitable ``execute``. ``raw.page`` exists. If ``conn`` is not in autocommit
        mode, the caller commits or rolls back afterwards for the deletion to take effect or be discarded.

    Args:
        conn: Connection the ``DELETE`` runs on.
        task_id: Value matched against the ``task_id`` column.

    Returns:
        The cursor's ``rowcount`` after the ``DELETE``: the number of rows deleted, ``0`` when no row matched, or
        whatever value the cursor reports when the count is unavailable.

    Raises:
        Exception: Whatever ``db_errors`` raises in place of, or lets through from, a failure in ``conn.execute``.
        UnboundLocalError: When ``db_errors`` suppresses an exception from ``conn.execute``, because ``cursor`` is
            then never assigned.

    """
    with db_errors():
        cursor = await conn.execute(_DISCARD, (task_id,))
    return cursor.rowcount
