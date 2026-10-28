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

from pipeline.db.pool import Connection
from pipeline.errors import db_errors


@dataclass(frozen=True, slots=True)
class Failure:
    """One validation rule a held row failed.

    Instances are frozen and use slots. ``hold`` stores each failure as a JSON object built with ``asdict``, whose
    keys are ``rule``, ``version``, ``message`` and ``until``, with ``until`` null for an error. No field is
    validated.

    Attributes:
        rule (str): Name of the rule that failed.
        version (int): Version of the rule that failed.
        message (str): Description of the failure.
        until (str | None): The job a waiting record waits for, or ``None`` for an error. Defaults to ``None``.

    """

    rule: str  # the rule's name, e.g. "order_id_present"
    version: int  # the rule's version when it ran
    message: str  # what was wrong with this record
    until: str | None = None  # the job a waiting record waits for; None for an error


@dataclass(frozen=True, slots=True, kw_only=True)
class HeldRow:
    """One record held back by validation, as stored in ``quarantine.row``.

    Instances are frozen, use slots, and take every field as a required keyword argument in the generated
    ``__init__``. ``__post_init__`` validates ``raw_page_no``, ``raw_index`` and ``failures``. The
    generated ``__eq__`` compares all fields and the generated ``__hash__`` hashes all fields, so ``hash`` raises
    ``TypeError`` because ``payload`` is a ``dict``.

    Attributes:
        task_id (int): Identifier of the task that produced the record. Not validated.
        connector (str): The connector the record was fetched from, e.g. ``"tiktok_shop"``. Not validated.
        job (str): Name of the job that produced the record. Not validated.
        source_id (UUID): The source (the account the task was for) the record belongs to. Not validated.
        partition_key (str): Partition the record would have been loaded into. Not validated.
        raw_page_no (int): With ``task_id``, the ``raw.page`` row the record came from, 0 or more.
        raw_index (int): Position of the record within that page, 0 or more.
        group_key (str | None): Key grouping related records, or ``None``. Not validated.
        payload (dict[str, Any]): The record itself. The dict passed in is stored as is, not copied.
        failures (tuple[Failure, ...]): The rules the record failed, at least one.

    """

    task_id: int  # control.task id of the task that held it
    connector: str
    job: str
    source_id: UUID
    partition_key: str
    raw_page_no: int  # with task_id, the raw.page row the record came from
    raw_index: int  # position of the record in that page
    group_key: str | None  # records held together share it, e.g. the order id
    payload: dict[str, Any]  # the record as landed
    failures: tuple[Failure, ...]  # every error rule it broke; never empty

    def __post_init__(self) -> None:
        """Validate ``raw_page_no``, ``raw_index`` and ``failures``.

        Checks run in order: ``raw_page_no`` and ``raw_index`` together, then ``failures``. No field is changed. No
        upper bound is checked, although the ``quarantine.row`` columns are ``integer``; the table's own ``CHECK``
        constraints repeat the ``0 or more`` rule.

        Preconditions:
            None.

        Returns:
            None.

        Raises:
            ValueError: If ``raw_page_no`` or ``raw_index`` is less than ``0``, or if ``failures`` is empty.

        """
        if self.raw_page_no < 0 or self.raw_index < 0:
            raise ValueError(f"raw_page_no and raw_index must be 0 or more, got {self.raw_page_no}, {self.raw_index}")
        if not self.failures:
            raise ValueError("a held row needs at least one failure")


#: ``COPY`` statement ``hold`` streams rows through, one per ``HeldRow``, its columns in ``HeldRow`` field order.
_COPY: Final = (
    "COPY quarantine.row (task_id, connector, job, source_id, partition_key, raw_page_no, raw_index, group_key, "
    "payload, failures) FROM STDIN"
)


async def hold(conn: Connection, rows: Sequence[HeldRow]) -> int:
    """Copy held rows into ``quarantine.row`` inside the caller's open transaction.

    Checks the connection's transaction status first, so the check applies even when ``rows`` is empty. For a
    non-empty ``rows``, opens a cursor and a ``COPY`` using ``_COPY`` and writes one record per row, with
    ``payload`` and ``failures`` wrapped as ``Jsonb`` (each failure converted with ``asdict``). Does not commit;
    the rows become visible only when the caller's transaction commits. The cursor and copy run inside
    ``db_errors``, so connection and conflict errors are translated and every other error propagates unchanged.
    ``HeldRow.__post_init__`` guarantees each row has at least one failure and non-negative positions, matching the
    table's ``CHECK`` constraints.

    Preconditions:
        ``conn`` is inside a transaction (``pq.TransactionStatus.INTRANS``). Every ``payload`` is serialisable by
        ``Jsonb``. No other coroutine is using ``conn`` for the duration of the call.

    Args:
        conn: The connection to copy through, already inside a transaction.
        rows: The rows to copy, in the order they are written.

    Returns:
        ``len(rows)``: the number of rows passed to the copy, not a count reported by the server. Returns ``0``
        without opening a cursor when ``rows`` is empty.

    Raises:
        RuntimeError: If ``conn.info.transaction_status`` is anything other than ``pq.TransactionStatus.INTRANS``
            (idle, in error, or busy), including when ``rows`` is empty.
        TypeError: If a ``payload`` value cannot be serialised to JSON; ``db_errors`` passes it through unchanged.
        DatabaseConflict: From ``db_errors``, on a serialization failure or deadlock during the cursor or copy.
        DatabaseUnavailable: From ``db_errors``, on a lost or refused connection during the cursor or copy.
        psycopg.Error: Any other database error during the cursor or copy, unchanged, for example a value outside
            the ``integer`` range of ``raw_page_no`` or ``raw_index``.

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
                        row.connector,
                        row.job,
                        row.source_id,
                        row.partition_key,
                        row.raw_page_no,
                        row.raw_index,
                        row.group_key,
                        Jsonb(row.payload),
                        Jsonb([asdict(failure) for failure in row.failures]),
                    )
                )
    return len(rows)
