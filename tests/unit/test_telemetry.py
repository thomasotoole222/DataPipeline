"""Unit tests for the OTLP log shipper, ``configure`` and ``bind_task`` in ``pipeline.telemetry``."""

import json
import logging
import os
import queue
import subprocess
import sys
import threading
from collections.abc import Callable
from typing import Any, cast

import httpx
import pytest
from opentelemetry.metrics import Counter
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


def _shipper(
    handler: Callable[[httpx.Request], httpx.Response], **kwargs: Any
) -> tuple[OtlpLogShipper, logging.Logger]:
    """Return a shipper posting to ``handler`` and a logger feeding it, not propagating to the root.

    Args:
        handler: Mock collector.
        **kwargs: Passed to ``OtlpLogShipper``.

    Returns:
        The shipper, not started, and its logger.

    """
    dropped = MeterProvider().get_meter("test").create_counter("dropped")
    shipper = OtlpLogShipper("http://collector", Resource.create({}), dropped, **kwargs)
    shipper._transport = httpx.MockTransport(handler)
    logger = logging.getLogger(f"test_telemetry.{len(logging.Logger.manager.loggerDict)}")
    logger.propagate = False
    logger.addHandler(_EagerQueueHandler(shipper.queue))
    return shipper, logger


def test_lone_surrogates_are_escaped() -> None:
    """Check a lone surrogate in the body, an attribute key or value is escaped and the batch still posts."""
    bodies: list[str] = []

    def collector(request: httpx.Request) -> httpx.Response:
        bodies.append(request.content.decode("utf-8"))
        return httpx.Response(200, json={})

    shipper, logger = _shipper(collector)
    shipper.start()
    bad = os.fsdecode(b"\xff")
    logger.warning("file %s", bad, extra={"pipeline_task": {f"key{bad}": f"value{bad}"}})
    shipper.stop(5)
    assert len(bodies) == 1
    record = json.loads(bodies[0])["resourceLogs"][0]["scopeLogs"][0]["logRecords"][0]
    assert record["body"]["stringValue"] == "file \\udcff"
    assert record["attributes"] == [{"key": "key\\udcff", "value": {"stringValue": "value\\udcff"}}]


class _Counted:
    """Stands in for an OpenTelemetry ``Counter`` and records each ``add``."""

    def __init__(self) -> None:
        self.added: list[tuple[int, dict[str, str]]] = []

    def add(self, amount: int, attributes: dict[str, str] | None = None) -> None:
        self.added.append((amount, dict(attributes or {})))


@pytest.mark.parametrize(
    ("rejected", "counted"),
    [(-3, []), (0, []), ("1", [(1, {"reason": "rejected"})]), (5, [(2, {"reason": "rejected"})])],
)
def test_rejected_count_is_clamped_to_the_batch(rejected: object, counted: list[tuple[int, dict[str, str]]]) -> None:
    """Check the collector's rejected count is clamped to between 0 and the records sent before it is counted."""
    dropped = _Counted()
    shipper = OtlpLogShipper("http://collector", Resource.create({}), cast(Counter, dropped))
    reply = {"partialSuccess": {"rejectedLogRecords": rejected}}
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=reply)))
    records = [
        logging.getLogger("test_telemetry").makeRecord(
            "test_telemetry", logging.WARNING, __file__, 1, "record", (), None
        )
        for _ in range(2)
    ]
    with client:
        shipper._ship(client, records)
    assert dropped.added == counted


def test_stop_keeps_a_thread_that_outlives_the_join() -> None:
    """Check ``stop`` keeps a still-running thread and ``start`` refuses while it is alive."""
    entered = threading.Event()
    release = threading.Event()

    def collector(request: httpx.Request) -> httpx.Response:
        entered.set()
        release.wait(10)
        return httpx.Response(200, json={})

    shipper, logger = _shipper(collector, max_wait=0.01)
    shipper.start()
    logger.warning("slow")
    assert entered.wait(5)
    shipper.stop(0)
    assert shipper._thread is not None
    with pytest.raises(RuntimeError, match="still stopping"):
        shipper.start()
    release.set()
    shipper.stop(5)
    assert shipper._thread is None
    shipper.start()
    shipper.stop(5)


def test_shutdown_posts_use_a_quarter_of_the_time_left() -> None:
    """Check a POST started during ``stop`` gives each httpx phase at most a quarter of the time left."""
    timeouts: list[dict[str, float]] = []

    def collector(request: httpx.Request) -> httpx.Response:
        timeouts.append(request.extensions["timeout"])
        return httpx.Response(200, json={})

    shipper, logger = _shipper(collector, max_wait=30)
    shipper.start()
    logger.warning("queued before stop")
    shipper.stop(2)
    assert len(timeouts) == 1
    assert all(0 < value <= 0.5 for value in timeouts[0].values())


def test_unprintable_message_still_ships() -> None:
    """Check a message whose ``__str__`` raises is shipped as ``<unprintable message: ExcType>``."""

    class Unprintable:
        def __str__(self) -> str:
            raise LookupError("no text")

    record = logging.makeLogRecord({"msg": Unprintable(), "args": None})
    assert _EagerQueueHandler(queue.SimpleQueue()).prepare(record).msg == "<unprintable message: LookupError>"


# Fails configure once after a global side effect, then calls it again, printing the second call's exception.
_CONFIGURE_TWICE = """
import os
import pipeline.telemetry as telemetry


def broken_start(self):
    raise OSError("no CA bundle")


telemetry.OtlpLogShipper.start = broken_start
try:
    telemetry.configure(env="local", worker_id="test-1", log_level="INFO")
except OSError:
    pass
try:
    telemetry.configure(env="local", worker_id="test-1", log_level="INFO")
except Exception as err:
    print(type(err).__name__, flush=True)
os._exit(0)
"""


def test_configure_runs_once_even_after_a_failure() -> None:
    """Check a second ``configure`` raises RuntimeError when the first failed after a global side effect."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("OTEL_")}
    env["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://127.0.0.1:9"
    done = subprocess.run(  # noqa: S603 - runs this interpreter on a fixed script
        [sys.executable, "-c", _CONFIGURE_TWICE], env=env, capture_output=True, text=True, timeout=60, check=True
    )
    assert done.stdout.strip().splitlines()[-1] == "RuntimeError"
