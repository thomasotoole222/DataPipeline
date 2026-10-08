"""OpenTelemetry setup for the pipeline: tracing, metrics and structured logging.

Call ``configure`` once at startup and ``Telemetry.shutdown`` before exit::

    telemetry = configure(env=env, worker_id=worker_id, log_level="info")
    try:
        with bind_task(task_id=..., run_id=..., attempt=..., job=..., shop_id=..., partition=...):
            run()
    finally:
        telemetry.shutdown()

What it provides:
    - ``configure`` builds the tracer and meter providers, instruments ``httpx``
      and ``psycopg``, and attaches logging handlers to the root logger.
    - ``bind_task`` binds a task's ``hub.*`` fields for the duration of a ``with`` block.
      Log records and ``traced`` spans created inside it carry them, as does the span
      current at entry.
    - ``traced`` wraps a function in a span and records exceptions with a
      ``hub.error_class`` attribute.
    - ``Instruments`` holds the pipeline's metric instruments.

Logs go to stderr as JSON lines (``JsonFormatter``) and, when an OTLP endpoint is
configured, to the collector through ``OtlpLogShipper``. The shipper is a small
custom OTLP/HTTP exporter that runs on its own thread, never retries, and counts
every record it fails to deliver in ``pipeline.log.dropped``.

Review notes:
    TODO(review): Shutdown timing is not bounded end to end. See
        ``OtlpLogShipper.stop`` and ``Telemetry.shutdown``.
"""

# STANDARD
import copy
import json
import logging
import logging.handlers
import os
import queue
import sys
import threading
import time
import traceback
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

# 3rd PARTY
import httpx
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.psycopg import PsycopgInstrumentor
from opentelemetry.metrics import Counter, Histogram
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode, Tracer

# LOCAL
from cipher_pipeline._wrapping import around

#: Deployment environment names.
type Env = Literal["local", "staging", "production"]
#: The bound task's ``hub.*`` fields.
type TaskAttributes = Mapping[str, str | int]

#: The bound task's fields; unset outside a task.
_task: ContextVar[TaskAttributes] = ContextVar("cipher_task")


@dataclass(frozen=True, slots=True)
class Instruments:
    """OpenTelemetry instruments for pipeline metrics.

    Attributes:
        task_duration: ``pipeline.task.duration``, wall-clock time per task attempt, in
            seconds, with ``job`` and ``outcome`` attributes.
        rows: ``pipeline.rows``, rows per stage, with ``job`` and a ``stage`` attribute of
            ``fetched``, ``quarantined`` or ``written``.
        source_requests: ``pipeline.source.requests``, every HTTP request made to a source
            API, with ``source``, ``status`` and ``result`` attributes.
        log_dropped: ``pipeline.log.dropped``, every log record not delivered to the
            collector, with a ``reason`` attribute: ``convert``, ``post_error``, ``http_status``,
            ``rejected``, ``shed``, ``shutdown`` or ``error``.

    """

    task_duration: Histogram  # pipeline.task.duration
    rows: Counter  # pipeline.rows
    source_requests: Counter  # pipeline.source.requests
    log_dropped: Counter  # pipeline.log.dropped


@contextmanager
def bind_task(
    *, task_id: str, run_id: str, attempt: int, job: str, shop_id: str, partition: str
) -> Iterator[None]:
    """Bind a task's ``hub.*`` fields to the current context for the duration of a ``with`` block.

    The fields are stored in the ``_task`` context variable. ``ContextFilter`` copies them
    onto every log record and ``traced`` adds them to each span it starts. They are also set
    on the span that is current at entry, if it is recording. Spans created by the ``httpx``
    and ``psycopg`` instrumentation do not get them, but they share the task's trace.

    On exit the context variable is restored to its previous value, so nested calls
    behave correctly.

    Preconditions:
        None. Works before ``configure``; the current span is then non-recording, so only
        ``_task`` is set.

    Args:
        task_id: Stored as ``hub.task_id``.
        run_id: Stored as ``hub.run_id``.
        attempt: Stored as ``hub.attempt``.
        job: Stored as ``hub.job``.
        shop_id: Stored as ``hub.shop_id``.
        partition: Stored as ``hub.partition``.

    Yields:
        None.

    Raises:
        Nothing itself. An exception raised in the ``with`` block propagates unchanged, and
        ``_task`` is restored even when one does.

    """
    attrs: TaskAttributes = {
        "hub.task_id": task_id,
        "hub.run_id": run_id,
        "hub.attempt": attempt,
        "hub.job": job,
        "hub.shop_id": shop_id,
        "hub.partition": partition,
    }
    token = _task.set(attrs)
    try:
        span = trace.get_current_span()
        if span.is_recording():
            span.set_attributes(attrs)
        yield
    finally:
        _task.reset(token)


def task_attributes() -> TaskAttributes:
    """Return the bound task's ``hub.*`` fields, or an empty dict outside a task.

    Preconditions:
        None.

    Returns:
        A new dict on every call. Mutating it does not change the bound task.

    Raises:
        Nothing.

    """
    return dict(_task.get({}))


def traced[F: Callable[..., Any]](name: str, **attrs: str) -> Callable[[F], F]:
    """Decorate a function so each call runs inside a span called ``name``.

    The span carries the current task's ``hub.*`` fields (see ``bind_task``) plus
    ``attrs``. Values in ``attrs`` take precedence if a key appears in both. Task fields
    are read when the span starts; outside ``bind_task`` the span has only ``attrs``.

    If the function raises an ``Exception``, the span records it, gets ``ERROR``
    status with its error class, and gets a ``hub.error_class`` attribute taken from
    the exception's ``error_class`` (or ``"Unexpected"`` if it has none). The full
    message stays in the exception event. Any other ``BaseException`` (for example
    ``CancelledError``) gives ``ERROR`` status with the class name and is not recorded.
    The exception is then re-raised.

    Preconditions:
        The decorated function is not a generator or async generator function.

    Args:
        name: Span name.
        **attrs: Static string attributes added to every span.

    Returns:
        A decorator. It returns a wrapper with the same signature as the decorated function.

    Raises:
        TypeError: At decoration, from ``around``, if the decorated function is a generator
            or async generator function.
        RuntimeError: At call time, from ``around``, if the span context manager swallows an
            exception. ``span`` always re-raises, so this should not occur.
        BaseException: At call time, whatever the wrapped function raises (including
            ``CancelledError`` and ``KeyboardInterrupt``), re-raised unchanged.

    """

    @contextmanager
    def span(_fn: Callable[..., Any], _args: tuple[Any, ...], _kwargs: dict[str, Any]) -> Iterator[None]:
        """Run one call of the wrapped function inside a span. Used by ``around``.

        See ``traced`` for the span's attributes and exception handling.

        Preconditions:
            Called only by ``around``, once per call of the wrapped function.

        Args:
            _fn: The wrapped function. Unused; ``around`` passes it positionally.
            _args: Positional arguments of the call. Unused.
            _kwargs: Keyword arguments of the call. Unused.

        Yields:
            None.

        Raises:
            Exception: Whatever the wrapped call raised, re-raised unchanged after it is recorded
                on the span.
            BaseException: Any other ``BaseException`` (for example ``CancelledError`` or
                ``KeyboardInterrupt``), re-raised unchanged after the span gets ``ERROR``
                status with the class name. It is not recorded and gets no ``hub.error_class``.

        """
        tracer = trace.get_tracer("cipher_pipeline")
        with tracer.start_as_current_span(
            name,
            attributes={**task_attributes(), **attrs},
            record_exception=False,
            set_status_on_exception=False,
        ) as active:
            try:
                yield
            except Exception as exc:
                named = getattr(exc, "error_class", None)
                error_class = named if isinstance(named, str) else "Unexpected"
                active.record_exception(exc)
                active.set_status(Status(StatusCode.ERROR, error_class))
                active.set_attribute("hub.error_class", error_class)
                raise
            except BaseException as exc:
                active.set_status(Status(StatusCode.ERROR, type(exc).__name__))
                raise

    return around(span)


class ContextFilter(logging.Filter):
    """Add trace and task context to log records. Never drops a record.

    Sets these attributes on each record:
        trace_id: Current trace ID as 32 lowercase hex characters.
        span_id: Current span ID as 16 lowercase hex characters.
        cipher_task: Dict of the current task's ``hub.*`` fields (see ``bind_task``),
            empty outside a task.

    ``trace_id`` and ``span_id`` are only set when there is a valid current span.

    Attach this to handlers, not loggers. Logger filters do not run on records
    propagated from child loggers.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Add ``trace_id``, ``span_id`` and ``cipher_task`` to ``record`` and keep it.

        ``trace_id`` and ``span_id`` are set only when there is a valid current span.
        ``cipher_task`` is always set to the current task's ``hub.*`` fields.

        Preconditions:
            None.

        Args:
            record: The record to enrich. Modified in place.

        Returns:
            Always ``True``. This filter enriches records and never drops them.

        Raises:
            Nothing.

        """
        span_context = trace.get_current_span().get_span_context()
        if span_context.is_valid:
            record.trace_id = format(span_context.trace_id, "032x")
            record.span_id = format(span_context.span_id, "016x")
        record.cipher_task = task_attributes()
        return True


class JsonFormatter(logging.Formatter):
    """Format each log record as a single line of JSON.

    Output keys:
        ts: Record creation time in UTC, ISO 8601 with milliseconds
            (for example ``2026-10-07T15:11:00.123+00:00``).
        level: Level name.
        logger: Logger name.
        msg: The formatted message.
        trace_id, span_id: Present only when set by ``ContextFilter`` and non-empty.
        hub.*: The task fields from ``record.cipher_task``, flattened into the top level.
        exc: Formatted traceback, present only when the record has exception info.
        stack: The record's ``stack_info``, present only when it was logged with
            ``stack_info=True``.

    Use with ``ContextFilter`` on the same handler, or the trace and task keys
    will be missing.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Return ``record`` as one line of JSON. See the class docstring for the keys.

        Preconditions:
            ``ContextFilter`` ran on the same handler. Otherwise the trace and task keys are
            absent.

        Args:
            record: The record to format. Not modified.

        Returns:
            One line of JSON with no newline.

        Raises:
            Exception: Whatever ``record.getMessage()`` raises when the format arguments do not
                match the message (for example ``TypeError``). ``logging.Handler.emit`` catches
                it and calls ``handleError``. ``json.dumps`` does not raise for values it cannot
                serialise, because of ``default=str``.

        """
        line: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key in ("trace_id", "span_id"):
            if value := getattr(record, key, None):
                line[key] = value
        line.update(getattr(record, "cipher_task", {}))
        if record.exc_info:
            line["exc"] = self.formatException(record.exc_info)
        elif record.exc_text:
            line["exc"] = record.exc_text
        if record.stack_info:
            line["stack"] = record.stack_info
        return json.dumps(line, default=str)

#: Maps Python log levels to OpenTelemetry severity numbers and text.
#: Each row is ``(minimum Python level, OTel SeverityNumber, severity text)``,
#: ordered highest first. A record takes the first row whose level it meets or
#: exceeds, and always gets the base number of that OTel range (for example
#: ERROR is always 17, never 18 to 20).
_SEVERITY = (
    (logging.CRITICAL, 21, "CRITICAL"),
    (logging.ERROR, 17, "ERROR"),
    (logging.WARNING, 13, "WARNING"),
    (logging.INFO, 9, "INFO"),
    (0, 5, "DEBUG"),
)
#: Maps ``str(float)`` output for non-finite values to the spellings used in
#: OTLP/JSON. Standard JSON cannot represent these values.
_NON_FINITE = {"nan": "NaN", "inf": "Infinity", "-inf": "-Infinity"}
#: Sentinel placed on the export queue to wake the worker thread. Compared by
#: identity, never exported.
_WAKE = logging.makeLogRecord({})
#: httpx timeout in seconds for each phase (connect, write, read, pool) of one export
#: POST. A slow POST can take longer than this in total.
_POST_TIMEOUT = 3.0
#: Seconds ``stop()`` waits for queued records to be sent. Records still queued at the
#: deadline are discarded and counted in ``pipeline.log.dropped``.
_FLUSH_TIMEOUT = 5.0
#: Backlog limit in batches: the queue holds at most ``max_batch * _BACKLOG_BATCHES``
#: records. Beyond that the oldest records are shed and counted in
#: ``pipeline.log.dropped``.
_BACKLOG_BATCHES = 10
#: Longest ``exception.stacktrace`` sent, in characters; longer traces keep their end.
_MAX_STACKTRACE = 32_768


class OtlpLogShipper:
    """Batch log records from a queue and POST them to an OTLP/HTTP collector.

    ``_EagerQueueHandler`` puts records on ``queue`` with ``msg`` already formatted and any
    exception stored in ``otel_exception``. A background daemon thread takes them off in
    batches of up to ``max_batch`` records, waiting at most ``max_wait`` seconds after the
    first record of a batch, and POSTs each batch to ``{endpoint}/v1/logs`` as an OTLP JSON
    export request. Records are grouped into one scope per logger name.

    Failed batches are not retried. Every record that is lost to a failed POST, a
    conversion error, a collector rejection, a full backlog, shutdown or an unexpected
    error is added to ``dropped``, with a ``reason`` attribute.

    Records logged on the shipper's own thread (for example by ``httpx``) are
    discarded to prevent a feedback loop. They are not counted as dropped.
    """

    def __init__(
        self, endpoint: str, resource: Resource, dropped: Counter, *, max_batch: int = 500, max_wait: float = 2.0
    ) -> None:
        """Create a shipper. Call ``start`` to begin exporting.

        Preconditions:
            ``endpoint`` is a base URL without ``/v1/logs``.

        Args:
            endpoint: Collector base URL, without ``/v1/logs``.
            resource: OTel resource whose attributes are sent with every batch.
            dropped: Counter incremented for every record that is not delivered.
            max_batch: Maximum records per POST. Must be at least 1. Also sets the
                backlog limit to ``max_batch * _BACKLOG_BATCHES`` records.
            max_wait: Maximum seconds to wait for a batch to fill after its first
                record arrives. Must be greater than 0.

        Returns:
            None.

        Raises:
            ValueError: If ``max_batch < 1``, or ``max_wait`` is not greater than 0
                (including NaN).

        """
        if max_batch < 1 or not max_wait > 0:
            raise ValueError("max_batch must be >= 1 and max_wait > 0")
        self._url = endpoint.rstrip("/") + "/v1/logs"
        self._resource = _attributes(resource.attributes)
        self._dropped = dropped
        self._max_batch = max_batch
        self._max_wait = max_wait
        self._queue: queue.SimpleQueue[logging.LogRecord] = queue.SimpleQueue()
        self._transport: httpx.BaseTransport | None = None  # test hook
        self._stopping = threading.Event()
        self._deadline = 0.0
        self._thread: threading.Thread | None = None

    @property
    def queue(self) -> queue.SimpleQueue[logging.LogRecord]:
        """The queue to pass to the ``QueueHandler`` that feeds this shipper.

        Preconditions:
            None.

        Returns:
            The same ``SimpleQueue`` on every call.

        Raises:
            Nothing.

        """
        return self._queue

    def start(self) -> None:
        """Start the background export thread. Does nothing if it is already running.

        The thread's HTTP client is uninstrumented so that exporting logs does not
        create spans.

        Preconditions:
            Called from one thread, not concurrently with ``start`` or ``stop``.

        Returns:
            None.

        Raises:
            OSError, ssl.SSLError: From ``httpx.Client`` if TLS setup fails (for example the CA
                bundle cannot be loaded).
            RuntimeError: If the OS cannot create the thread. The HTTP client is closed and the
                shipper stays stopped, so ``start`` can be retried.

        """
        if self._thread is not None and self._thread.is_alive():
            return
        self._stopping.clear()
        client = httpx.Client(timeout=httpx.Timeout(_POST_TIMEOUT), transport=self._transport)
        HTTPXClientInstrumentor.uninstrument_client(client)
        thread = threading.Thread(target=self._run, args=(client,), name="otlp-log-shipper", daemon=True)
        try:
            thread.start()
        except BaseException:
            client.close()
            raise
        self._thread = thread

    def stop(self) -> None:
        """Export queued records, then stop the background thread.

        Waits up to ``_FLUSH_TIMEOUT`` seconds. Records still queued when time runs
        out are discarded and counted as dropped. Does nothing if not started.

        Preconditions:
            Called from one thread, not concurrently with ``start``. The queue handler is
            already detached, so nothing new is enqueued.

        Returns:
            None. If ``join`` times out it returns with the thread still running, after
            logging a WARNING through this module's logger.

        Raises:
            Nothing.

        """
        thread = self._thread
        if thread is None:
            return
        self._thread = None
        self._deadline = time.monotonic() + _FLUSH_TIMEOUT
        self._stopping.set()
        self._queue.put(_WAKE)
        thread.join(timeout=_FLUSH_TIMEOUT)
        if thread.is_alive():
            logging.getLogger(__name__).warning(
                "log shipper did not stop within %.1fs; records still queued are dropped", _FLUSH_TIMEOUT
            )

    def _run(self, client: httpx.Client) -> None:
        """Body of the export thread. Closes ``client`` when it returns.

        Collects and ships batches until ``_stopping`` is set, then drains the
        queue until the deadline leaves less than ``_POST_TIMEOUT`` seconds, then
        discards and counts anything left.

        Preconditions:
            Runs on the shipper thread only, as the target started by ``start``. It compares
            record threads with ``threading.get_ident()`` to skip its own records.

        Args:
            client: Uninstrumented HTTP client used for every POST.

        Returns:
            None.

        Raises:
            Nothing to the caller: it runs on its own thread. Each pass catches ``Exception``,
            writes one line naming it to stderr, and counts its batch as dropped with reason
            ``error``. An exception from the final ``_shed`` or ``client.close()`` would end
            the thread, which is already finishing, and is printed by ``threading.excepthook``.

        """
        own = threading.get_ident()
        try:
            while not self._stopping.is_set():
                batch: list[logging.LogRecord] = []
                try:
                    self._collect(own, batch)
                    self._ship(client, batch)
                    batch.clear()
                    self._shed()
                except Exception as exc:
                    print(f"otlp-log-shipper: {type(exc).__name__}: {exc}", file=sys.stderr)
                    self._dropped.add(len(batch), {"reason": "error"})
                    self._stopping.wait(0.1)
            while time.monotonic() < self._deadline - _POST_TIMEOUT:
                rest: list[logging.LogRecord] = []
                try:
                    self._take(own, rest)
                    if not rest:
                        break
                    self._ship(client, rest)
                except Exception as exc:
                    print(f"otlp-log-shipper: {type(exc).__name__}: {exc}", file=sys.stderr)
                    self._dropped.add(len(rest), {"reason": "error"})
            self._shed(keep=0, reason="shutdown")
        finally:
            client.close()

    def _collect(self, own: int, batch: list[logging.LogRecord]) -> None:
        """Block until a record arrives, then fill ``batch`` for up to ``max_wait`` seconds.

        Returns early with an empty batch if ``_WAKE`` arrives first, and stops
        filling if ``_WAKE`` arrives later. Records from thread ``own`` are skipped.

        Preconditions:
            Runs on the shipper thread only. ``batch`` is empty.

        Args:
            own: Thread ID of the shipper thread.
            batch: List to fill, in queue order.

        Returns:
            None. ``batch`` then holds up to ``max_batch`` records.

        Raises:
            Nothing.

        """
        while True:
            first = self._queue.get()
            if first is _WAKE:
                return
            if first.thread != own:
                break
        batch.append(first)
        end = time.monotonic() + self._max_wait
        while len(batch) < self._max_batch:
            remaining = end - time.monotonic()
            if remaining <= 0:
                break
            try:
                record = self._queue.get(timeout=remaining)
            except queue.Empty:
                break
            if record is _WAKE:
                break
            if record.thread != own:
                batch.append(record)

    def _take(self, own: int, batch: list[logging.LogRecord]) -> None:
        """Fill ``batch`` with up to ``max_batch`` records without blocking.

        Skips ``_WAKE`` and records from thread ``own``. Used while draining at
        shutdown.

        Preconditions:
            Runs on the shipper thread only. ``batch`` is empty.

        Args:
            own: Thread ID of the shipper thread.
            batch: List to fill, in queue order.

        Returns:
            None. ``batch`` then holds up to ``max_batch`` records, or none if the queue
            was empty.

        Raises:
            Nothing.

        """
        while len(batch) < self._max_batch:
            try:
                record = self._queue.get_nowait()
            except queue.Empty:
                break
            if record is not _WAKE and record.thread != own:
                batch.append(record)

    def _shed(self, keep: int | None = None, *, reason: str = "shed") -> None:
        """Discard the oldest queued records until at most ``keep`` remain.

        Discarded records are counted as dropped, except ``_WAKE`` and records from
        the shipper's own thread.

        Preconditions:
            Runs on the shipper thread only. It compares record threads with
            ``threading.get_ident()``.

        Args:
            keep: Records to keep. Defaults to ``max_batch * _BACKLOG_BATCHES``.
                Pass 0 to discard everything at shutdown.
            reason: ``reason`` attribute for the dropped count.

        Returns:
            None.

        Raises:
            Nothing.

        """
        limit = self._max_batch * _BACKLOG_BATCHES if keep is None else keep
        own = threading.get_ident()
        shed = 0
        while self._queue.qsize() > limit:
            try:
                record = self._queue.get_nowait()
            except queue.Empty:
                break
            if record is not _WAKE and record.thread != own:
                shed += 1
        if shed:
            self._dropped.add(shed, {"reason": reason})

    def _ship(self, client: httpx.Client, batch: list[logging.LogRecord]) -> None:
        """Convert ``batch`` to OTLP and POST it in one request.

        Records that fail to convert are dropped individually (reason ``convert``). If
        the POST raises (``post_error``) or returns a non-2xx status (``http_status``),
        every converted record is counted as dropped. On a 2xx response, records the
        collector reports as rejected in ``partialSuccess`` are counted (``rejected``).

        Preconditions:
            Runs on the shipper thread only.

        Args:
            client: HTTP client to POST with.
            batch: Records to send. Not modified.

        Returns:
            None.

        Raises:
            Nothing: every ``Exception`` from converting a record or from the POST is caught
            and the affected records are counted as dropped.

        """
        scopes: dict[str, list[dict[str, Any]]] = {}
        built = 0
        for record in batch:
            try:
                scopes.setdefault(record.name, []).append(_log_record(record))
                built += 1
            except Exception:
                self._dropped.add(1, {"reason": "convert"})
        if not built:
            return
        try:
            response = client.post(self._url, json=_export_request(self._resource, scopes))
        except Exception:
            self._dropped.add(built, {"reason": "post_error"})
        else:
            if not response.is_success:
                self._dropped.add(built, {"reason": "http_status"})
            else:
                rejected = _rejected_count(response)
                if rejected:
                    self._dropped.add(rejected, {"reason": "rejected"})


def _rejected_count(response: httpx.Response) -> int:
    """Return how many log records the collector rejected in a 2xx export response.

    Reads ``partialSuccess.rejectedLogRecords`` from the OTLP/JSON response body. The
    count may be a string, as OTLP/JSON encodes int64 values that way.

    Preconditions:
        ``response`` is a 2xx response to an OTLP/HTTP logs export, with its body read.

    Args:
        response: The collector's response.

    Returns:
        The rejected count, or 0 when the body is empty, not JSON, not an object, has no
        ``partialSuccess`` or no ``rejectedLogRecords``, or the count is not an integer.

    Raises:
        Nothing: every ``Exception`` while parsing is caught and treated as 0.

    """
    try:
        partial = response.json().get("partialSuccess") or {}
        return int(partial.get("rejectedLogRecords") or 0)
    except Exception:  # empty or non-JSON body: nothing rejected
        return 0


def _export_request(resource: list[dict[str, Any]], scopes: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Build an OTLP/JSON ``ExportLogsServiceRequest`` body.

    Produces one ``resourceLogs`` entry with one ``scopeLogs`` entry per key in
    ``scopes``. The scope name is the key, which ``_ship`` sets to the logger name.

    Preconditions:
        ``scopes`` is non-empty. ``_ship`` guarantees it.

    Args:
        resource: Resource attributes, already converted to OTLP ``KeyValue`` form
            by ``_attributes``.
        scopes: OTLP log records, already converted by ``_log_record``, keyed by
            scope name.

    Returns:
        A dict ready to be serialised as the POST body to ``/v1/logs``. The input
        lists are referenced, not copied.

    Raises:
        Nothing.

    """
    return {
        "resourceLogs": [
            {
                "resource": {"attributes": resource},
                "scopeLogs": [{"scope": {"name": name}, "logRecords": records} for name, records in scopes.items()],
            }
        ]
    }


def _log_record(record: logging.LogRecord) -> dict[str, Any]:
    """Convert a log record to an OTLP/JSON ``LogRecord``.

    Sets:
        timeUnixNano: ``record.created`` in nanoseconds, as a string.
        severityNumber, severityText: From the first ``_SEVERITY`` row whose
            minimum level ``record.levelno`` meets.
        body: ``record.msg``, already formatted by ``_EagerQueueHandler``.
        traceId, spanId: Copied from the record when both were set by
            ``ContextFilter``.
        attributes: The ``cipher_task`` fields merged with ``otel_exception``, with
            exception fields taking precedence on a key collision, plus
            ``code.stacktrace`` from ``record.stack_info`` when it is set. Omitted when
            all are empty.

    Preconditions:
        ``record`` was prepared by ``_EagerQueueHandler``.

    Args:
        record: The prepared record. Not modified.

    Returns:
        A new dict.

    Raises:
        StopIteration: If ``record.levelno`` is negative, since no ``_SEVERITY``
            row matches. ``_ship`` catches this and counts the record as dropped.
        Exception: Whatever ``_attributes`` raises for a ``cipher_task`` or
            ``otel_exception`` value. ``_ship`` catches it and counts the record as dropped.

    """
    number, text = next((num, name) for floor, num, name in _SEVERITY if record.levelno >= floor)
    out: dict[str, Any] = {
        "timeUnixNano": str(round(record.created * 1_000_000_000)),
        "severityNumber": number,
        "severityText": text,
        "body": {"stringValue": str(record.msg)},
    }
    trace_id = getattr(record, "trace_id", None)
    span_id = getattr(record, "span_id", None)
    if isinstance(trace_id, str) and isinstance(span_id, str):
        out["traceId"] = trace_id
        out["spanId"] = span_id
    task = getattr(record, "cipher_task", None)
    exception = getattr(record, "otel_exception", None)
    attributes = {
        **(task if isinstance(task, Mapping) else {}),
        **(exception if isinstance(exception, Mapping) else {}),
    }
    if isinstance(record.stack_info, str):
        attributes["code.stacktrace"] = record.stack_info
    if attributes:
        out["attributes"] = _attributes(attributes)
    return out


def _exception_attributes(record: logging.LogRecord) -> dict[str, str]:
    """Return OTel semantic-convention exception attributes for ``record``.

    Returns an empty dict if the record has no exception. Otherwise returns:
        exception.type: Fully qualified class name, without the module for
            built-in exceptions (for example ``ValueError`` but
            ``httpx.ConnectError``).
        exception.message: ``str(exc)``, or ``<unprintable {type}>`` if that raises.
        exception.stacktrace: The formatted traceback, including chained exceptions.
            A trace longer than ``_MAX_STACKTRACE`` characters keeps its last
            ``_MAX_STACKTRACE`` characters (the most recent frames and the exception
            line) after a ``[truncated N chars]`` line.

    Called by ``_EagerQueueHandler.prepare`` on the original record, on the logging thread,
    before ``exc_info`` is cleared on the queued copy.

    Preconditions:
        ``record`` is the original record, with ``exc_info`` intact.

    Args:
        record: The record to read. Not modified.

    Returns:
        A new dict, empty when there is no exception.

    Raises:
        Nothing: ``str(exc)`` is guarded, and ``traceback.format_exception`` reports an
        unprintable exception instead of raising.

    """
    exc = record.exc_info[1] if record.exc_info else None
    if exc is None:
        return {}
    cls = type(exc)
    name = cls.__qualname__ if cls.__module__ == "builtins" else f"{cls.__module__}.{cls.__qualname__}"
    try:
        message = str(exc)
    except Exception:
        message = f"<unprintable {name}>"
    stack = "".join(traceback.format_exception(exc))
    if len(stack) > _MAX_STACKTRACE:
        stack = f"[truncated {len(stack) - _MAX_STACKTRACE} chars]\n" + stack[-_MAX_STACKTRACE:]
    return {
        "exception.type": name,
        "exception.message": message,
        "exception.stacktrace": stack,
    }


def _attributes(values: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Convert a mapping to a list of OTLP/JSON ``KeyValue`` entries.

    Each value is converted with ``_any_value``. Order follows the mapping.

    Preconditions:
        None.

    Args:
        values: Attribute names and values.

    Returns:
        A new list with one ``KeyValue`` dict per item.

    Raises:
        Exception: Whatever ``_any_value`` raises for a value.

    """
    return [{"key": key, "value": _any_value(value)} for key, value in values.items()]


def _any_value(value: object) -> dict[str, Any]:
    """Convert a Python value to an OTLP/JSON ``AnyValue``.

    Conversions:
        bool: ``boolValue``. Checked before ``int``, since ``bool`` subclasses it.
        int: ``intValue``, as a decimal string as OTLP/JSON requires for int64.
        float: ``doubleValue``. NaN and infinities become the strings ``"NaN"``,
            ``"Infinity"`` and ``"-Infinity"`` via ``_NON_FINITE``.
        list, tuple: ``arrayValue``, converting each item recursively.
        Anything else: ``stringValue`` using ``str(value)``.

    Preconditions:
        None.

    Args:
        value: The value to convert.

    Returns:
        A new ``AnyValue`` dict.

    Raises:
        RecursionError: For a list or tuple that contains itself, or is nested too deeply.
        Exception: Whatever ``str(value)`` raises in the fallback case.

    """
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": _NON_FINITE.get(repr(value), value)}
    if isinstance(value, list | tuple):
        return {"arrayValue": {"values": [_any_value(item) for item in value]}}
    return {"stringValue": str(value)}


@dataclass(frozen=True, slots=True)
class Telemetry:
    """Handles to the tracing, metrics and logging set up for the pipeline.

    Call ``shutdown`` once at process exit to flush and detach everything.

    Attributes:
        tracer: Tracer for pipeline spans.
        instruments: Metric instruments (see ``Instruments``).
        tracer_provider: Provider that owns span export.
        meter_provider: Provider that owns metric export.
        log_shipper: Ships logs to the collector. ``None`` when no collector is
            configured, for example in development.
        queue_handler: Root logger handler that feeds ``log_shipper``. ``None``
            when ``log_shipper`` is ``None``.
        stderr_handler: Root logger handler that writes JSON lines to stderr. ``configure``
            always attaches it.

    """

    tracer: Tracer
    instruments: Instruments
    tracer_provider: TracerProvider
    meter_provider: MeterProvider
    log_shipper: OtlpLogShipper | None
    queue_handler: logging.Handler | None = None
    stderr_handler: logging.Handler | None = None
    _once: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False, compare=False)

    def shutdown(self) -> None:
        """Flush and shut down all telemetry. Only the first call has any effect.

        In order:
            1. Detach ``queue_handler`` from the root logger, so no new records
               reach the shipper.
            2. Stop ``log_shipper``, exporting queued records (see
               ``OtlpLogShipper.stop`` for the time limit).
            3. Remove the ``httpx`` and ``psycopg`` instrumentation process-wide.
            4. Shut down the tracer provider, exporting pending spans.
            5. Shut down the meter provider, exporting final metric values,
               including drops counted while stopping the shipper.
            6. Detach ``stderr_handler``.

        Each step runs through ``_attempt``, so a failure in one does not stop the
        rest. Logs produced during steps 2 to 5 go to stderr only.

        Preconditions:
            None. Safe to call more than once; later calls return at once.

        Returns:
            None.

        Raises:
            Nothing for ``Exception``: each step runs in ``_attempt``, which logs it and
            continues.
            BaseException: For example ``KeyboardInterrupt``, propagates from the step it
                interrupts and skips the remaining steps. Later calls do not retry them.

        Review notes:
            TODO(review): Total time is unbounded from here. The shipper waits up
                to ``_FLUSH_TIMEOUT``, but provider shutdown uses the exporters'
                own timeouts, which default to 30 seconds or more each. Container
                stop grace periods are often 10 to 30 seconds.

        """
        if not self._once.acquire(blocking=False):
            return
        root = logging.getLogger()
        if (queue_handler := self.queue_handler) is not None:
            _attempt("detach queue handler", lambda: root.removeHandler(queue_handler))
        if self.log_shipper is not None:
            _attempt("stop log shipper", self.log_shipper.stop)
        _attempt("uninstrument httpx", HTTPXClientInstrumentor().uninstrument)
        _attempt("uninstrument psycopg", PsycopgInstrumentor().uninstrument)
        _attempt("shut down tracer provider", self.tracer_provider.shutdown)
        _attempt("shut down meter provider", self.meter_provider.shutdown)
        if (stderr_handler := self.stderr_handler) is not None:
            _attempt("detach stderr handler", lambda: root.removeHandler(stderr_handler))


def _attempt(step: str, action: Callable[[], object]) -> None:
    """Run one shutdown step, logging any exception instead of raising it.

    Used by ``Telemetry.shutdown`` so that one failing step does not prevent the
    rest from running. The return value of ``action`` is ignored.

    Preconditions:
        None.

    Args:
        step: Short description used in the error message.
        action: Zero-argument callable to run.

    Returns:
        None.

    Raises:
        Nothing for ``Exception``: it is caught and logged at ERROR with its traceback.
        BaseException: Propagates from ``action``.

    Notes:
        The failure is logged through ``logging``, so where it ends up depends on
        which handlers are still attached at that step. Fine in practice: before
        step 1 it goes to the shipper, after it to stderr, and after step 6 to
        ``logging.lastResort`` at ERROR level.

    """
    try:
        action()
    except Exception:
        logging.getLogger(__name__).exception("telemetry shutdown: %s failed", step)


class _EagerQueueHandler(logging.handlers.QueueHandler):
    """Queue handler that prepares records for ``OtlpLogShipper``.

    Unlike the stock ``QueueHandler``, it does not run a formatter. Instead it
    merges the message arguments into ``msg`` and stores the exception as OTel
    attributes in ``otel_exception``, on the logging thread, before the record is
    queued. This captures argument values as they were when the call was made,
    rather than whenever the shipper thread gets to them.

    Attach ``ContextFilter`` to this handler. Filters run before ``prepare``, so
    the trace and task fields are copied onto the queued record.
    """

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        """Return a copy of ``record`` ready to cross to the shipper thread.

        The copy has:
            msg: The formatted message. If formatting fails, ``"{msg} (args: {args!r})"``,
                or ``str(record.msg)`` if that fails too.
            args: ``None``.
            otel_exception: Exception attributes from ``_exception_attributes``,
                or an empty dict.
            exc_info, exc_text: ``None``, so traceback objects and their frames
                are not held in the queue.

        The original record is not modified, so other handlers still see its
        arguments and exception.

        Preconditions:
            Called by ``QueueHandler.emit`` on the logging thread, after the handler's
            filters have run.

        Args:
            record: The original record. Not modified.

        Returns:
            A shallow copy of ``record``, changed as listed above.

        Raises:
            Nothing: formatting is guarded, and ``_exception_attributes`` does not raise.

        """
        out = copy.copy(record)
        try:
            out.msg = record.getMessage()
        except Exception:
            try:
                out.msg = f"{record.msg} (args: {record.args!r})"
            except Exception:
                out.msg = str(record.msg)
        out.args = None
        out.otel_exception = _exception_attributes(record)
        out.exc_info = None
        out.exc_text = None
        return out


#: Serialises ``configure``.
_lock = threading.Lock()
#: The configured ``Telemetry``; set once.
_active: Telemetry | None = None


def configure(*, env: Env, worker_id: str, log_level: str) -> Telemetry:
    """Set up tracing, metrics and logging for this process and return the handles.

    Creates a tracer provider and meter provider, registers them as the global
    providers, instruments ``httpx`` and ``psycopg``, and attaches handlers to the
    root logger:
        - A JSON stderr handler, always.
        - A queue handler feeding an ``OtlpLogShipper``, when an OTLP endpoint is set.

    The OTLP endpoint comes from ``OTEL_EXPORTER_OTLP_ENDPOINT``. It is required
    unless ``env`` is ``"local"``. Without it, spans and metrics are created but not
    exported, and logs go to stderr only.

    Resource attributes:
        service.name: ``OTEL_SERVICE_NAME``, or ``pipeline-worker``.
        service.version: ``RAILWAY_GIT_COMMIT_SHA``, or ``dev``.
        deployment.environment.name: ``env``.
        service.instance.id: ``worker_id``. Must be unique per running worker.

    The root logger is set to ``log_level``. The ``httpx`` and ``httpcore`` loggers
    are set to WARNING to suppress per-request logs.

    Can only be called once per process. Call ``Telemetry.shutdown`` before exit;
    nothing is registered with ``atexit``.

    Preconditions:
        Called once per process, before any ``traced`` call whose spans should be
        exported.

    Args:
        env: Deployment environment.
        worker_id: Unique identifier for this worker instance.
        log_level: Root log level name, case-insensitive (for example ``"info"``).

    Returns:
        The configured ``Telemetry``.

    Raises:
        RuntimeError: If called more than once, or if ``OTEL_EXPORTER_OTLP_ENDPOINT`` is
            missing when ``env`` is not ``"local"``.
        ValueError: If ``log_level`` is not a known level name.
        BaseException: Anything raised while registering providers, instrumenting or
            attaching handlers is re-raised after ``Telemetry.shutdown`` has cleaned up.
            Building the resource, exporters, providers and shipper before that is not
            expected to raise: malformed ``OTEL_*`` values are logged and replaced by their
            defaults (checked 2026-10-08).

    """
    global _active
    with _lock:
        if _active is not None:
            raise RuntimeError("telemetry is already configured; configure() runs once per process")
        level = logging.getLevelNamesMapping().get(log_level.upper())
        if level is None:
            raise ValueError(f"unknown log level {log_level!r}")
        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip().rstrip("/")
        if not endpoint and env != "local":
            raise RuntimeError(f"OTEL_EXPORTER_OTLP_ENDPOINT is required when env is {env!r}")
        resource = Resource.create(
            {
                "service.name": os.environ.get("OTEL_SERVICE_NAME") or "pipeline-worker",
                "service.version": os.environ.get("RAILWAY_GIT_COMMIT_SHA") or "dev",
                "deployment.environment.name": env,
                "service.instance.id": worker_id,
            }
        )
        span_exporter = OTLPSpanExporter() if endpoint else None
        metric_exporter = OTLPMetricExporter() if endpoint else None
        tracer_provider = TracerProvider(resource=resource)
        if span_exporter is not None:
            tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[PeriodicExportingMetricReader(metric_exporter)] if metric_exporter is not None else [],
        )
        meter = meter_provider.get_meter("cipher_pipeline")
        instruments = Instruments(
            task_duration=meter.create_histogram("pipeline.task.duration", unit="s"),
            rows=meter.create_counter("pipeline.rows", unit="{row}"),
            source_requests=meter.create_counter("pipeline.source.requests", unit="{request}"),
            log_dropped=meter.create_counter("pipeline.log.dropped", unit="{record}"),
        )
        shipper = OtlpLogShipper(endpoint, resource, instruments.log_dropped) if endpoint else None
        queue_handler: logging.Handler | None = None
        if shipper is not None:
            queue_handler = _EagerQueueHandler(shipper.queue)
            queue_handler.addFilter(ContextFilter())
        stderr_handler = logging.StreamHandler()
        stderr_handler.setFormatter(JsonFormatter())
        stderr_handler.addFilter(ContextFilter())
        telemetry = Telemetry(
            tracer=tracer_provider.get_tracer("cipher_pipeline"),
            instruments=instruments,
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            log_shipper=shipper,
            queue_handler=queue_handler,
            stderr_handler=stderr_handler,
        )
        try:
            trace.set_tracer_provider(tracer_provider)
            metrics.set_meter_provider(meter_provider)
            HTTPXClientInstrumentor().instrument(tracer_provider=tracer_provider, meter_provider=meter_provider)
            PsycopgInstrumentor().instrument(tracer_provider=tracer_provider, meter_provider=meter_provider)
            for noisy in ("httpx", "httpcore"):
                logging.getLogger(noisy).setLevel(logging.WARNING)
            root = logging.getLogger()
            root.setLevel(level)
            if shipper is not None:
                shipper.start()
            if queue_handler is not None:
                root.addHandler(queue_handler)
            root.addHandler(stderr_handler)
        except BaseException:
            telemetry.shutdown()
            raise
        _active = telemetry
        return telemetry
