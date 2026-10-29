"""Unit tests for the error type a failure report carries: ``TaskReport`` and ``worker.loop._failure_report``."""

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import pytest

from pipeline.db.control import ClaimedTask, TaskReport
from pipeline.errors import Outcome
from pipeline.worker.loop import _failure_report

NOW = datetime(2026, 10, 29, 12, tzinfo=UTC)


def report(**kw: Any) -> TaskReport:
    """Build a failed report, overriding any field."""
    base: dict[str, Any] = {
        "outcome": Outcome.UNEXPECTED,
        "error_class": "Unexpected",
        "error_type": "RecursionError",
        "error_message": "boom",
        "retry_after": None,
        "rows_fetched": 0,
        "rows_written": 0,
        "rows_rejected": 0,
        "duration": timedelta(0),
    }
    base.update(kw)
    return TaskReport(**base)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"error_type": None}, "error_type must be None exactly when the outcome is succeeded"),
        ({"error_type": ""}, "error_type must not be empty"),
        (
            {"outcome": Outcome.SUCCEEDED, "error_class": None, "error_message": None},
            "error_type must be None exactly when the outcome is succeeded",
        ),
    ],
    ids=["failed without a type", "empty type", "succeeded with a type"],
)
def test_task_report_checks_the_error_type(fields: dict[str, Any], message: str) -> None:
    """Check a report has an error type exactly when it is not a success, and never an empty one."""
    with pytest.raises(ValueError, match=message):
        report(**fields)


def test_task_report_makes_the_error_type_storable() -> None:
    """Check a NUL in the error type is stored as U+FFFD."""
    assert report(error_type="Bad\x00Type").error_type == "Bad�Type"


def test_failure_report_names_the_exception_beside_its_class() -> None:
    """Check an unclassified exception is reported as Unexpected with its own type name as the error type."""
    task = ClaimedTask(
        id=1,
        run_id=1,
        connector="tiktok_shop",
        job="orders",
        source_id=UUID(int=1),
        partition_key="2026-10-19",
        attempt=1,
        lease_token=UUID(int=2),
        created_at=NOW,
        not_before=NOW,
        claimed_at=NOW,
        lease_expires_at=NOW + timedelta(minutes=5),
        trace_context=None,
    )
    built = _failure_report(task, RecursionError("maximum recursion depth exceeded"))
    assert (built.outcome, built.error_class, built.error_type) == (Outcome.UNEXPECTED, "Unexpected", "RecursionError")
