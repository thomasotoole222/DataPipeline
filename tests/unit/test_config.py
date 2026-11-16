"""Unit tests for the ``raw_retention``, ``health_port`` and ``archive_`` settings of ``pipeline.config.Settings``."""

from datetime import timedelta

import pytest
from pydantic import ValidationError

from pipeline.config import Settings

URL = "postgresql://user@localhost/warehouse"
ARCHIVE = {
    "archive_bucket": "raw-archive",
    "archive_endpoint": "https://storage.example",
    "archive_region": "auto",
    "archive_access_key_id": "key",
    "archive_secret_access_key": "s3cr3t-value",
}


@pytest.fixture(autouse=True)
def no_pipeline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear the raw retention, health port and archive variables, so only the test's own value is read."""
    monkeypatch.delenv("PIPELINE_RAW_RETENTION", raising=False)
    monkeypatch.delenv("PIPELINE_HEALTH_PORT", raising=False)
    for name in ARCHIVE:
        monkeypatch.delenv(f"PIPELINE_{name.upper()}", raising=False)


def test_raw_retention_defaults_to_7_days() -> None:
    """Check the raw retention is 7 days when not set."""
    assert Settings(warehouse_url=URL, env="local", _env_file=None).raw_retention == timedelta(days=7)


def test_raw_retention_reads_an_iso_8601_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check PIPELINE_RAW_RETENTION is read as an ISO-8601 duration."""
    monkeypatch.setenv("PIPELINE_RAW_RETENTION", "PT2H30M")
    assert Settings(warehouse_url=URL, env="local", _env_file=None).raw_retention == timedelta(hours=2, minutes=30)


def test_raw_retention_must_be_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check a zero retention is refused."""
    monkeypatch.setenv("PIPELINE_RAW_RETENTION", "PT0S")
    with pytest.raises(ValidationError):
        Settings(warehouse_url=URL, env="local", _env_file=None)


def test_health_port_defaults_to_8080() -> None:
    """Check /health listens on 8080 when the port is not set."""
    assert Settings(warehouse_url=URL, env="local", _env_file=None).health_port == 8080


def test_health_port_reads_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check PIPELINE_HEALTH_PORT sets the port."""
    monkeypatch.setenv("PIPELINE_HEALTH_PORT", "9090")
    assert Settings(warehouse_url=URL, env="local", _env_file=None).health_port == 9090


@pytest.mark.parametrize("port", ["0", "65536", "http"])
def test_health_port_must_be_a_tcp_port(monkeypatch: pytest.MonkeyPatch, port: str) -> None:
    """Check a port outside 1 to 65535, or not a number, is refused."""
    monkeypatch.setenv("PIPELINE_HEALTH_PORT", port)
    with pytest.raises(ValidationError):
        Settings(warehouse_url=URL, env="local", _env_file=None)


def test_archive_is_off_by_default() -> None:
    """Check a worker with no archive settings has no archive bucket."""
    assert Settings(warehouse_url=URL, env="local", _env_file=None).archive_bucket is None


def test_archive_reads_all_five_settings() -> None:
    """Check the five archive settings are taken together, the keys masked."""
    settings = Settings(warehouse_url=URL, env="local", _env_file=None, **ARCHIVE)
    assert settings.archive_bucket == "raw-archive"
    assert settings.archive_secret_access_key is not None
    assert "s3cr3t-value" not in repr(settings)


@pytest.mark.parametrize("missing", sorted(ARCHIVE))
def test_archive_refuses_a_partial_store(missing: str) -> None:
    """Check leaving out any one archive setting is refused, and the message names it."""
    values = {name: value for name, value in ARCHIVE.items() if name != missing}
    with pytest.raises(ValidationError, match=missing):
        Settings(warehouse_url=URL, env="local", _env_file=None, **values)
