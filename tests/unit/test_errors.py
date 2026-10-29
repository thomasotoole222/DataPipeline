"""Unit tests for the warehouse error types and ``db_errors`` in ``pipeline.errors``."""

import psycopg
import psycopg_pool
import pytest

from pipeline.errors import (
    DatabaseConflict,
    DatabaseUnavailable,
    KeyViolation,
    Outcome,
    Overflow,
    PipelineError,
    TooDeep,
    classify,
    db_errors,
)

UNAVAILABLE: list[type[Exception]] = [
    psycopg.OperationalError,
    psycopg.errors.ConnectionException,
    psycopg.errors.ConnectionFailure,
    psycopg.errors.SqlclientUnableToEstablishSqlconnection,
    psycopg.errors.ConnectionDoesNotExist,
    psycopg.errors.AdminShutdown,
    psycopg.errors.CrashShutdown,
    psycopg.errors.CannotConnectNow,
    psycopg_pool.PoolTimeout,
]

CONFLICT: list[type[Exception]] = [
    psycopg.errors.SerializationFailure,
    psycopg.errors.DeadlockDetected,
]

PASS_THROUGH: list[type[Exception]] = [
    psycopg.errors.QueryCanceled,
    psycopg.errors.UndefinedTable,
    psycopg.InterfaceError,
    psycopg.errors.OutOfMemory,
    psycopg.errors.SqlserverRejectedEstablishmentOfSqlconnection,
    psycopg.errors.ProtocolViolation,
    psycopg.errors.InvalidPassword,
    ValueError,
]


@pytest.mark.parametrize("cls", [DatabaseUnavailable, DatabaseConflict])
def test_classify_database_errors_as_retryable(cls: type[PipelineError]) -> None:
    """Classify each warehouse error as retryable under its own class name."""
    assert classify(cls("x")) == (Outcome.RETRYABLE, cls.__name__)


def _translate(original: Exception, expected: type[PipelineError]) -> None:
    """Raise ``original`` inside ``db_errors`` and check it becomes ``expected``."""
    with pytest.raises(expected) as info, db_errors():
        raise original
    assert info.value.__cause__ is original
    assert str(info.value).startswith(f"warehouse: {type(original).__name__}: ")
    assert classify(info.value) == (Outcome.RETRYABLE, expected.__name__)


@pytest.mark.parametrize("cls", UNAVAILABLE, ids=lambda c: c.__name__)
def test_db_errors_raises_unavailable(cls: type[Exception]) -> None:
    """Turn connection-level errors into ``DatabaseUnavailable``."""
    _translate(cls("boom"), DatabaseUnavailable)


@pytest.mark.parametrize("cls", CONFLICT, ids=lambda c: c.__name__)
def test_db_errors_raises_conflict(cls: type[Exception]) -> None:
    """Turn serialization failures and deadlocks into ``DatabaseConflict``."""
    _translate(cls("boom"), DatabaseConflict)


@pytest.mark.parametrize("cls", PASS_THROUGH, ids=lambda c: c.__name__)
def test_db_errors_passes_other_errors_through(cls: type[Exception]) -> None:
    """Re-raise every other error as the same object."""
    original = cls("boom")
    with pytest.raises(cls) as info, db_errors():
        raise original
    assert info.value is original


def test_db_errors_raises_key_violation_without_the_key_values() -> None:
    """Turn a unique violation into a permanent ``KeyViolation`` whose message leaves out Postgres's detail."""
    original = psycopg.errors.UniqueViolation("duplicate key value violates unique constraint\nDETAIL: Key (id)=(o1)")
    with pytest.raises(KeyViolation) as info, db_errors():
        raise original
    assert info.value.__cause__ is original
    assert str(info.value).startswith("warehouse: unique key ")
    assert "o1" not in str(info.value)
    assert classify(info.value) == (Outcome.FAILED, "KeyViolation")


@pytest.mark.parametrize("cls", [KeyViolation, Overflow, TooDeep], ids=lambda c: c.__name__)
def test_deterministic_errors_are_permanent(cls: type[PipelineError]) -> None:
    """Classify each error the same input always reproduces as failed, so it is not retried."""
    assert classify(cls("x")) == (Outcome.FAILED, cls.__name__)
