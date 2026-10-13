from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, Final
from uuid import UUID

import psycopg
from psycopg import sql
from psycopg.rows import class_row

from cipher_pipeline.config import Settings
from cipher_pipeline.db.pool import Connection
from cipher_pipeline.errors import LeaseConflict, Outcome, db_errors
from cipher_pipeline.timeutil import ensure_utc

TASK_CHANNEL: Final = "control_task"  # NOTIFY channel the orchestrator signals on a new runnable task


@dataclass(frozen=True, slots=True, kw_only=True)
class ClaimedTask:
    id: int
    run_id: int
    job: str
    shop_id: UUID
    partition_key: str
    attempt: int
    lease_token: UUID
    created_at: datetime
    not_before: datetime
    claimed_at: datetime
    trace_context: str | None

    def __post_init__(self) -> None:
        for name in ("created_at", "not_before", "claimed_at"):
            object.__setattr__(self, name, ensure_utc(getattr(self, name)))


_MESSAGE_LIMIT: Final = 2000


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskReport:
    outcome: Outcome
    error_class: str | None
    error_message: str | None
    retry_after: timedelta | None
    rows_fetched: int | None
    rows_written: int
    rows_rejected: int
    duration: timedelta

    def __post_init__(self) -> None:
        if (self.outcome == Outcome.SUCCEEDED) != (self.error_class is None):
            raise ValueError(f"error_class must be None exactly when the outcome is succeeded, got {self.outcome}")
        if min(self.rows_fetched, self.rows_written, self.rows_rejected) < 0:
            raise ValueError("rows_fetched must be >= 0")
        if self.duration < timedelta(0):
            raise ValueError(f"duration must be >= 0, got {self.duration}")
        if self.error_message is not None and len(self.error_message) > _MESSAGE_LIMIT:
            object.__setattr__(self, "error_message", self.error_message[:_MESSAGE_LIMIT])


class LeaseLoss(StrEnum):
    HEARTBEAT = "heartbeat"
    SILENCE = "silence"
    COMMIT = "commit"


_CLAIM: Final = "SELECT * FROM control.claim(%s)"


async def claim(conn: Connection, worker_id: str) -> ClaimedTask | None:
    with db_errors():
        async with conn.cursor(row_factory=class_row(ClaimedTask)) as cursor:
            await cursor.execute(_CLAIM, (worker_id,))
            return await cursor.fetchone()


_HEARTBEAT: Final = "SELECT control.heartbeat(%s, %s)"


async def hearbeat(conn: Connection, task: ClaimedTask) -> bool:
    with db_errors():
        cursor = await conn.execute(_HEARTBEAT, (task.id, task.lease_token))
        row = await cursor.fetchone()
    return row is not None and row[0] is True


_LEASE_CONFLICT_SQLSTATE: Final = "CH001"


@contextmanager
def _fenced(task: ClaimedTask) -> Iterator[None]:
    try:
        with db_errors():
            yield
    except psycopg.Error as exc:
        if exc.sqlstate != _LEASE_CONFLICT_SQLSTATE:
            raise
        raise LeaseConflict(f"fast {task.id}: lease {task.lease_token} is no longer held") from exc


_FENCE: Final = "SELECT control.fence(%s, %s)"


async def fence(conn: Connection, task: ClaimedTask) -> None:
    with _fenced(task):
        await conn.execute(_FENCE, (task.id, task.lease_token))


_REPORT: Final = "SELECT control.report_outcome(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"


async def report_outcome(conn: Connection, task: ClaimedTask, report: TaskReport) -> str:
    params = (
        task.id,
        task.lease_token,
        report.outcome.value,
        report.error_class,
        report.error_message,
        report.retry_after,
        report.rows_written,
        report.rows_rejected,
        report.duration // timedelta(milliseconds=1),
    )
    with _fenced(task):
        cursor = await conn.execute(_REPORT, params)
        row = await cursor.fetchone()
    if row is None:
        raise RuntimeError("control.report_outcome returned no row")
    return str(row[0])


_LEASE_LOSS: Final = "SELECT control.record_lease_loss(%s,%s,%s.%s.%s)"


async def record_lease_loss(
    conn: Connection, task: ClaimedTask, kind: LeaseLoss, worker_id: str, unwind: timedelta | None
) -> None:
    with db_errors():
        unwind_ms = None if unwind is None else unwind // timedelta(milliseconds=1)
        await conn.execute(_LEASE_LOSS, (task.id, task.lease_token, kind.value, worker_id, unwind_ms))


async def open_listener(settins: Settings) -> psycopg.AsyncConnection[[Any]]:
    with db_errors():
        listener = await psycopg.AsyncConnection.connect(
            Settings.warehouse_url.get_secret_value(), autocommit=True, application_name=f"{Settings.worker_id}-listen"
        )
        try:
            await listener.execute(sql.SQL("LISTEN {}").format(sql.Identifier(TASK_CHANNEL)))
        except BaseException:
            await listener.close()
            raise
    return listener


async def wait_for_task(listener: psycopg.AsyncConnection[Any], wait: timedelta) -> bool:
    received = False
    with db_errors():
        async for _ in listener.notifies(timeout=wait.total_seconds(), stop_after=1):
            received = True
    return received
