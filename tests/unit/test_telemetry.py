"""Unit tests for the OTLP log headers and ``bind_task`` in ``pipeline.telemetry``."""

import json
import logging
import os
import subprocess
import sys

import httpx
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource

from pipeline.telemetry import OtlpLogShipper, _EagerQueueHandler, bind_task, task_attributes

# Runs configure in a fresh interpreter (it runs once per process) and prints the headers the log shipper got.
_CONFIGURE = """
import json, os
import pipeline.telemetry as telemetry

seen = []


class Recording(telemetry.OtlpLogShipper):
    def __init__(self, *args, headers=None, **kwargs):
        seen.append(dict(headers or {}))
        super().__init__(*args, headers=headers, **kwargs)


telemetry.OtlpLogShipper = Recording
telemetry.configure(env="local", worker_id="test-1", log_level="INFO")
print(json.dumps(seen), flush=True)
os._exit(0)  # skip shutdown: there is no collector to flush to
"""


def _shipper_headers(**env: str) -> list[dict[str, str]]:
    """Return the headers ``configure`` passes to the log shipper, run with ``env`` added to a clean environment.

    Args:
        **env: ``OTEL_*`` variables to set for the run.

    Returns:
        One headers dict per shipper ``configure`` built.

    """
    base = {k: v for k, v in os.environ.items() if not k.startswith("OTEL_")}
    base["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://127.0.0.1:9"
    done = subprocess.run(  # noqa: S603 - runs this interpreter on a fixed script
        [sys.executable, "-c", _CONFIGURE], env=base | env, capture_output=True, text=True, timeout=60, check=True
    )
    seen: list[dict[str, str]] = json.loads(done.stdout.strip().splitlines()[-1])
    return seen


def test_configured_headers() -> None:
    """Check every POST goes to ``<endpoint>/v1/logs`` carrying the headers the shipper was built with."""
    posts: list[httpx.Request] = []

    def collector(request: httpx.Request) -> httpx.Response:
        posts.append(request)
        return httpx.Response(200, json={})

    dropped = MeterProvider().get_meter("test").create_counter("dropped")
    shipper = OtlpLogShipper("http://collector", Resource.create({}), dropped, headers={"authorization": "Bearer t"})
    shipper._transport = httpx.MockTransport(collector)
    logger = logging.getLogger("test_configured_headers")
    logger.propagate = False
    logger.addHandler(_EagerQueueHandler(shipper.queue))
    shipper.start()
    logger.warning("one record")
    shipper.stop(5)
    assert posts
    assert all(str(p.url) == "http://collector/v1/logs" for p in posts)
    assert all(p.headers["authorization"] == "Bearer t" for p in posts)


def test_logs_headers_take_precedence() -> None:
    """Check ``OTEL_EXPORTER_OTLP_LOGS_HEADERS`` wins over ``OTEL_EXPORTER_OTLP_HEADERS``, which applies alone."""
    general = "authorization=Bearer%20general,x-other=1"
    assert _shipper_headers(
        OTEL_EXPORTER_OTLP_LOGS_HEADERS="authorization=Bearer%20logs", OTEL_EXPORTER_OTLP_HEADERS=general
    ) == [{"authorization": "Bearer logs"}]
    assert _shipper_headers(OTEL_EXPORTER_OTLP_HEADERS=general) == [{"authorization": "Bearer general", "x-other": "1"}]


def test_bind_task_binds_the_connector() -> None:
    """Check ``pipeline.connector`` is bound inside the block and gone after it."""
    with bind_task(
        task_id="1", run_id="2", attempt=1, connector="tiktok_shop", job="orders", source_id="s", partition="p"
    ):
        assert task_attributes()["pipeline.connector"] == "tiktok_shop"
    assert task_attributes() == {}
