"""The ``raw`` schema: page validation, table naming, and saving, reading and discarding fetched pages.

Public API:
    raw_table: Build the ``raw`` schema identifier for a raw table name.
    RawPage: One page of results fetched for a task, as stored in a raw table.
    save_page: Insert a ``RawPage`` row, or overwrite the existing row with the same ``task_id`` and ``page_no``.
    last_page: Return the highest-numbered page stored for a task, or ``None``.
    read_pages: Stream the pages of one task as ``RawPage`` objects, ordered by ``page_no``.
    discard_pages: Delete every row of a task from a raw table and return the reported row count.
"""

import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

from psycopg import sql
from psycopg.rows import class_row
from psycopg.types.json import Jsonb

from cipher_pipeline.db.pool import Connection
from cipher_pipeline.errors import db_errors
from cipher_pipeline.timeutil import ensure_utc

_TABLE_NAME: Final = re.compile(r"raw\.([a-z][a-z0-9_]*)")


def raw_table(name: str) -> sql.Identifier:
    """Build the ``raw`` schema identifier for a raw table name.

    Matches the whole of ``name`` against ``_TABLE_NAME`` and pairs ``"raw"`` with the first capture group of the
    match to form a two-part ``sql.Identifier``.

    Preconditions:
        ``name`` is a ``str``. ``_TABLE_NAME`` is a compiled ``str`` pattern with at least one capture group, and
        group 1 always takes part in a full match with a non-empty value.

    Args:
        name: The table name to validate, matched in full against ``_TABLE_NAME``.

    Returns:
        ``sql.Identifier("raw", g)``, where ``g`` is group 1 of the match, naming table ``g`` in schema ``raw``.

    Raises:
        ValueError: If ``name`` does not fully match ``_TABLE_NAME``. The message includes ``repr(name)``.

    """
    match = _TABLE_NAME.fullmatch(name)
    if match is None:
        raise ValueError(f"not a raw table name: {name!r}")
    return sql.Identifier("raw", match.group(1))


@dataclass(frozen=True, slots=True, kw_only=True)
class RawPage:
    """One page of results fetched for a task, as stored in the raw page table.

    Instances are frozen, use slots, and take every field as a required keyword argument in the generated
    ``__init__``. ``__post_init__`` validates ``page_no`` and ``payload`` and replaces ``fetched_at`` with the result
    of ``ensure_utc``. The generated ``__eq__`` compares all fields and the generated ``__hash__`` hashes all fields.
    A dataclass subclass must also be frozen.

    Attributes:
        task_id (int): Identifier of the task the page belongs to. Not validated.
        page_no (int): Position of the page within the task, 0 or more.
        cursor_in (str | None): Cursor sent to fetch this page, or ``None``. Not validated.
        cursor_next (str | None): Cursor returned for the next page, or ``None``. Not validated.
        fetched_at (datetime): Time the page was fetched, as returned by ``ensure_utc``.
        payload (dict[str, Any]): Decoded page body. The dict passed in is stored as is, not copied.

    """

    task_id: int
    page_no: int
    cursor_in: str | None
    cursor_next: str | None
    fetched_at: datetime
    payload: dict[str, Any]

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


#: Comma-separated column list of the raw page table, in ``RawPage`` field order.
_COLUMNS: Final = sql.SQL("task_id, page_no, cursor_in, cursor_next, fetched_at, payload")

#: Upsert template for one ``RawPage`` row. Must be formatted with ``table`` and ``columns`` and executed with six
#: parameters in ``_COLUMNS`` order. On a ``(task_id, page_no)`` conflict, every other column is overwritten. The
#: ``ON CONFLICT`` clause requires a unique constraint or index on ``(task_id, page_no)`` in the target table.

#: Upsert statement template for one raw page row. ``{table}`` and ``{columns}`` are filled by ``sql.SQL.format``.
#: The six ``%s`` parameters are, in order, ``task_id``, ``page_no``, ``cursor_in``, ``cursor_next``, ``fetched_at``
#: and ``payload``. On a conflict on ``(task_id, page_no)`` it overwrites ``cursor_in``, ``cursor_next``,
#: ``fetched_at`` and ``payload`` with the incoming values.

_SAVE: Final = sql.SQL(
    "INSERT INTO {table} ({columns}) VALUES (%s, %s, %s, %s, %s, %s) "
    "ON CONFLICT (task_id, page_no) DO UPDATE SET cursor_in = EXCLUDED.cursor_in, "
    "cursor_next = EXCLUDED.cursor_next, fetched_at = EXCLUDED.fetched_at, payload = EXCLUDED.payload"
)


async def save_page(conn: Connection, table: str, page: RawPage) -> None:
    """Upsert one raw page into the raw table for ``table``.

    Builds the statement from ``_SAVE`` with ``raw_table(table)`` and ``_COLUMNS``, then executes it on ``conn`` inside
    ``db_errors()``. An existing row with the same ``task_id`` and ``page_no`` has its ``cursor_in``, ``cursor_next``,
    ``fetched_at`` and ``payload`` overwritten. ``payload`` is wrapped in ``Jsonb``. The function does not commit.

    Preconditions:
        ``conn`` is an open asynchronous connection that is not running another statement. The table named by
        ``raw_table(table)`` exists, has the columns in ``_COLUMNS``, and has a unique constraint or unique index on
        ``(task_id, page_no)``. ``page.payload`` is JSON-serialisable. The caller commits afterwards unless ``conn``
        is in autocommit mode.

    Args:
        conn: Connection the statement runs on.
        table: Logical table name passed to ``raw_table``.
        page: Page whose ``task_id``, ``page_no``, ``cursor_in``, ``cursor_next``, ``fetched_at`` and ``payload`` are
            written.

    Returns:
        None.

    Raises:
        Whatever ``db_errors`` raises or lets through when ``conn.execute`` fails, for example on a missing table, a
        missing ``(task_id, page_no)`` unique constraint or a payload that cannot be serialised; conditional on
        ``db_errors``. Any exception from ``raw_table`` or ``_SAVE.format`` propagates unchanged.

    """
    query = _SAVE.format(table=raw_table(table), columns=_COLUMNS)
    params = (page.task_id, page.page_no, page.cursor_in, page.cursor_next, page.fetched_at, Jsonb(page.payload))
    with db_errors():
        await conn.execute(query, params)


#: Query template that selects the row with the highest ``page_no`` for one ``task_id``. ``{columns}`` and ``{table}``
#: are filled by ``sql.SQL.format``. ``%s`` is the ``task_id`` parameter, bound at execution.

_LAST: Final = sql.SQL("SELECT {columns} FROM {table} WHERE task_id = %s ORDER BY page_no DESC LIMIT 1")


async def last_page(conn: Connection, table: str, task_id: int) -> RawPage | None:
    """Return the highest-numbered page row stored for a task.

    Builds the query from ``_LAST`` using ``raw_table(table)`` and ``_COLUMNS``. It then opens a cursor on ``conn``
    whose row factory is ``class_row(RawPage)``, executes the query with ``task_id`` bound, and fetches one row.
    Execution and fetching run inside ``db_errors()``. Building the query runs outside it.

    Preconditions:
        ``conn`` is an open asynchronous connection that no other coroutine is using concurrently. ``table`` is a
        name accepted by ``raw_table``. ``_COLUMNS`` names exactly the fields that ``RawPage`` accepts as keyword
        arguments.

    Args:
        conn: Connection to run the query on.
        table: Logical table name, passed to ``raw_table``.
        task_id: Task whose last page is fetched.

    Returns:
        The ``RawPage`` built from the row with the highest ``page_no`` for ``task_id`` (see ``_LAST`` for how
        ``NULL`` and tied values are ordered). ``None`` if the table has no row for ``task_id``, and also ``None`` if
        ``db_errors`` suppresses an exception.

    Raises:
        Any exception raised by ``raw_table`` or ``_LAST.format``, not passed through ``db_errors``. Exceptions
        raised while opening the cursor, executing or fetching, as translated or re-raised by ``db_errors``. This
        function swallows nothing itself. Whether ``db_errors`` swallows anything depends on its implementation.

    """
    query = _LAST.format(table=raw_table(table), columns=_COLUMNS)
    with db_errors():
        async with conn.cursor(row_factory=class_row(RawPage)) as cursor:
            await cursor.execute(query, (task_id,))
            return await cursor.fetchone()


#: SQL template that selects ``{columns}`` from ``{table}`` for one task, ordered by ``page_no`` ascending.
#: ``{columns}`` and ``{table}`` are filled in with ``sql.SQL.format``; the ``%s`` placeholder takes the task id.

_READ: Final = sql.SQL("SELECT {columns} FROM {table} WHERE task_id = %s ORDER BY page_no")


async def read_pages(conn: Connection, table: str, task_id: int) -> AsyncIterator[RawPage]:
    """Read the pages of one task from a raw table.

    Nothing runs until iteration starts. On the first step, all matching rows are fetched with a single cursor on
    ``conn`` inside ``db_errors()``, converted to ``RawPage`` by ``class_row``, and the cursor is closed; the pages are
    then yielded from memory.

    Preconditions:
        ``conn`` is open and not used concurrently until the first page has been yielded.
        ``table`` is a name that ``raw_table`` accepts.
        ``RawPage`` accepts the columns listed in ``_COLUMNS`` as keyword arguments.

    Args:
        conn: Connection the cursor is opened on.
        table: Logical table name, passed to ``raw_table``.
        task_id: Task whose rows are selected.

    Yields:
        One ``RawPage`` per row whose ``task_id`` matches, in ascending ``page_no`` order. Yields nothing if no row
        matches, so an unknown task is not distinguished from a task with no pages.

    Raises:
        Whatever ``db_errors`` translates database errors into, from executing the query or fetching rows. Errors
        from ``raw_table`` (raised before ``db_errors`` is entered) and from ``RawPage`` that ``db_errors`` does not
        handle propagate unchanged. All exceptions surface on the first iteration step, never at the call itself.
    """
    query = _READ.format(table=raw_table(table), columns=_COLUMNS)
    with db_errors():
        async with conn.cursor(row_factory=class_row(RawPage)) as cursor:
            await cursor.execute(query, (task_id,))
            pages = await cursor.fetchall()
    for page in pages:
        yield page


#: Statement template that deletes every row matching one ``task_id``. ``{table}`` is filled by ``sql.SQL.format``
#: and ``%s`` is bound to the task id when the statement is executed.
_DISCARD: Final = sql.SQL("DELETE FROM {table} WHERE task_id = %s")


async def discard_pages(conn: Connection, table: str, task_id: int) -> int:
    """Delete every row in ``table`` whose ``task_id`` column equals ``task_id``.

    Fills ``_DISCARD`` with ``raw_table(table)``, executes the statement on ``conn`` inside ``db_errors()``, and
    returns the cursor's ``rowcount``. The function does not commit and does not close the cursor.

    Preconditions:
        ``conn`` is open and accepts an awaitable ``execute``. ``table`` names an existing table that has a
        ``task_id`` column. If ``conn`` is not in autocommit mode, the caller commits or rolls back afterwards for
        the deletion to take effect or be discarded. ``table`` comes from a trusted source unless ``raw_table``
        quotes it as an identifier.

    Args:
        conn: Connection the ``DELETE`` runs on.
        table: Name of the table to delete from, passed through ``raw_table``.
        task_id: Value matched against the ``task_id`` column.

    Returns:
        The cursor's ``rowcount`` after the ``DELETE``: the number of rows deleted, ``0`` when no row matched, or
        whatever value the cursor reports when the count is unavailable.

    Raises:
        Exception: Any exception from ``raw_table`` or ``_DISCARD.format``, unwrapped, because both run before
            ``db_errors()`` is entered.
        Exception: Whatever ``db_errors`` raises in place of, or lets through from, a failure in ``conn.execute``.
        UnboundLocalError: When ``db_errors`` suppresses an exception from ``conn.execute``, because ``cursor`` is
            then never assigned.

    """
    query = _DISCARD.format(table=raw_table(table))
    with db_errors():
        cursor = await conn.execute(query, (task_id,))
    return cursor.rowcount
