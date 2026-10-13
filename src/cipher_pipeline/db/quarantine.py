"""The quarantine hold table: rows that failed validation and the bulk copy that holds them.

Public API:
    Failure: One validation rule a held row failed.
    HeldRow: One record held back by validation, as stored in ``quarantine.row``.
    hold: Copy held rows into ``quarantine.row`` inside the caller's open transaction.
"""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final
from uuid import UUID

from psycopg import pq
from psycopg.types.json import Jsonb

from cipher_pipeline.db.pool import Connection
from cipher_pipeline.db.raw import raw_table
from cipher_pipeline.errors import db_errors


@dataclass(frozen=True, slots=True)
class Failure:
    """One validation rule a held row failed.

    Instances are frozen and use slots. ``hold`` stores each failure as a JSON object built with ``asdict``. No field
    is validated.

    Attributes:
        rule (str): Name of the rule that failed.
        version (int): Version of the rule that failed.
        message (str): Description of the failure.

    """

    rule: str
    version: int
    message: str


@dataclass(frozen=True, slots=True, kw_only=True)
class HeldRow:
    """One record held back by validation, as stored in ``quarantine.row``.

    Instances are frozen, use slots, and take every field as a required keyword argument in the generated
    ``__init__``. ``__post_init__`` validates ``raw_table``, ``raw_page_no``, ``raw_index`` and ``failures``. The
    generated ``__eq__`` compares all fields and the generated ``__hash__`` hashes all fields, so ``hash`` raises
    ``TypeError`` because ``payload`` is a ``dict``.

    Attributes:
        task_id (int): Identifier of the task that produced the record. Not validated.
        job (str): Name of the job that produced the record. Not validated.
        shop_id (UUID): Shop the record belongs to. Not validated.
        partition_key (str): Partition the record would have been loaded into. Not validated.
        raw_table (str): Raw table the record was read from, in the form ``raw_table`` accepts.
        raw_page_no (int): Page of ``raw_table`` the record came from, 0 or more.
        raw_index (int): Position of the record within that page, 0 or more.
        group_key (str | None): Key grouping related records, or ``None``. Not validated.
        payload (dict[str, Any]): The record itself. The dict passed in is stored as is, not copied.
        failures (tuple[Failure, ...]): The rules the record failed, at least one.

    """

    task_id: int
    job: str
    shop_id: UUID
    partition_key: str
    raw_table: str
    raw_page_no: int
    raw_index: int
    group_key: str | None
    payload: dict[str, Any]
    failures: tuple[Failure, ...]

    def __post_init__(self) -> None:
        """Validate ``raw_table``, ``raw_page_no``, ``raw_index`` and ``failures``.

        Checks run in order: ``raw_table`` through ``raw_table``, then ``raw_page_no`` and ``raw_index`` together,
        then ``failures``. No field is changed.

        Preconditions:
            ``raw_page_no`` and ``raw_index`` support ``<`` against ``0``.

        Returns:
            None.

        Raises:
            ValueError: If ``raw_table`` is not a name ``raw_table`` accepts, if ``raw_page_no`` or ``raw_index`` is
                less than ``0``, or if ``failures`` is empty.
            TypeError: If ``raw_table`` is not a ``str``, or if ``raw_page_no`` or ``raw_index`` cannot be compared
                with ``0`` (for example ``None``); both come with an unrelated message.

        """
        raw_table(self.raw_table)
        if self.raw_page_no < 0 or self.raw_index < 0:
            raise ValueError(f"raw_page_no and raw_index must be 0 or more, got {self.raw_page_no}, {self.raw_index}")
        if not self.failures:
            raise ValueError("a held row needs at least one failure")


_COPY: Final = (
    "COPY quarantine.row (task_id, job, shop_id, partition_key, raw_table, raw_page_no, raw_index, group_key, "
    "payload, failures) FROM STDIN"
)


async def hold(conn: Connection, rows: Sequence[HeldRow]) -> int:
    """Copy held rows into the hold table inside the caller's open transaction.

    Checks the connection's transaction status first, so the check applies even when ``rows`` is empty. For a
    non-empty ``rows``, opens a cursor and a ``COPY`` using ``_COPY`` and writes one record per row, with
    ``payload`` and ``failures`` wrapped as ``Jsonb`` (each failure converted with ``asdict``). Does not commit;
    the rows become visible only when the caller's transaction commits. Database errors raised during the copy pass
    through ``db_errors``.

    Preconditions:
        ``conn`` is an async psycopg connection whose transaction status is ``pq.TransactionStatus.INTRANS``.
        Every element of ``rows`` has the attributes ``_COPY`` expects, in the same order. Every ``payload`` and
        every field of every item in ``failures`` is serialisable by ``Jsonb``, and every item in ``failures`` is a
        dataclass instance. No other coroutine is using ``conn`` for the duration of the call.

    Args:
        conn: The connection to copy through, already inside a transaction.
        rows: The rows to copy, in the order they are written.

    Returns:
        ``len(rows)``: the number of rows passed to the copy, not a count reported by the server. Returns ``0``
        without opening a cursor when ``rows`` is empty.

    Raises:
        RuntimeError: If ``conn.info.transaction_status`` is anything other than ``pq.TransactionStatus.INTRANS``,
            including when ``rows`` is empty.
        TypeError: If ``rows`` does not support ``len``, if an item in ``failures`` is not a dataclass instance, or
            if a value cannot be serialised to JSON. Whether ``db_errors`` translates these is unknown.
        Exception: Whatever ``db_errors`` raises for a database error during the cursor or copy, conditional on
            ``db_errors``.

    """
    if conn.info.transaction_status != pq.TransactionStatus.INTRANS:
        raise RuntimeError("hold must run inside the Load transaction")
    if not rows:
        return 0
    with db_errors():
        async with conn.cursor() as cursor, cursor.copy(_COPY) as copy:
            for row in rows:
                await copy.write_row(
                    (
                        row.task_id,
                        row.job,
                        row.shop_id,
                        row.partition_key,
                        row.raw_table,
                        row.raw_page_no,
                        row.raw_index,
                        row.group_key,
                        Jsonb(row.payload),
                        Jsonb([asdict(failure) for failure in row.failures]),
                    )
                )
    return len(rows)
