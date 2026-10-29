"""Unit tests for the ``raw_retention`` setting of ``pipeline.config.Settings``."""

from datetime import timedelta

import pytest
from pydantic import ValidationError

from pipeline.config import Settings

URL = "postgresql://user@localhost/warehouse"


@pytest.fixture(autouse=True)
def no_pipeline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear the raw retention variable, so only the test's own value is read."""
    monkeypatch.delenv("PIPELINE_RAW_RETENTION", raising=False)


def test_raw_retention_defaults_to_60_hours() -> None:
    """Check the raw retention is 60 hours when not set."""
    assert Settings(warehouse_url=URL, env="local", _env_file=None).raw_retention == timedelta(hours=60)


def test_raw_retention_reads_an_iso_8601_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check PIPELINE_RAW_RETENTION is read as an ISO-8601 duration."""
    monkeypatch.setenv("PIPELINE_RAW_RETENTION", "PT2H30M")
    assert Settings(warehouse_url=URL, env="local", _env_file=None).raw_retention == timedelta(hours=2, minutes=30)


def test_raw_retention_must_be_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check a zero retention is refused."""
    monkeypatch.setenv("PIPELINE_RAW_RETENTION", "PT0S")
    with pytest.raises(ValidationError):
        Settings(warehouse_url=URL, env="local", _env_file=None)
