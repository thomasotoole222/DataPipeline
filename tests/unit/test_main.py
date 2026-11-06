"""Unit tests for how ``pipeline.__main__`` starts the worker from its settings."""

from typing import Any, cast

import httpx
import pytest

import pipeline.__main__ as entry
from pipeline.config import Settings
from pipeline.registry import Registry
from pipeline.telemetry import Instruments

URL = "postgresql://user@localhost/warehouse"


async def test_serve_listens_on_the_configured_health_port(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check ``_serve`` starts ``/health`` on ``Settings.health_port``."""
    seen: dict[str, Any] = {}

    async def fake_run_worker(settings: Settings, instruments: Instruments, **kwargs: Any) -> None:
        seen.update(kwargs)

    def no_jobs(settings: Settings, http: httpx.AsyncClient, instruments: Instruments) -> Registry:
        return Registry()

    monkeypatch.setattr(entry, "run_worker", fake_run_worker)
    settings = Settings(warehouse_url=URL, env="local", health_port=9191, _env_file=None)
    await entry._serve(settings, cast(Instruments, None), no_jobs)
    assert seen["health_port"] == 9191
