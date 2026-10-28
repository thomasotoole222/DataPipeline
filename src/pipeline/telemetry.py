"""OpenTelemetry setup for the pipeline: tracing, metrics and structured logging.

Call ``configure`` once at startup and ``Telemetry.shutdown`` before exit::

    telemetry = configure(env=env, worker_id=worker_id, log_level="info")
    try:
        with bind_task(task_id=..., run_id=..., attempt=..., connector=..., job=..., source_id=..., partition=...):
            run()
    finally:
        telemetry.shutdown()

What it provides:
    - ``configure`` builds the tracer and meter providers, instruments ``httpx``
      and ``psycopg``, and attaches logging handlers to the root logger.
    - ``bind_task`` binds a task's ``pipeline.*`` fields for the duration of a ``with`` block.
      Log records and ``traced`` spans created inside it carry them, as does the span
      current at entry.
    - ``traced`` wraps a function in a span and records exceptions with a
      ``pipeline.error_class`` attribute.
    - ``Instruments`` holds the pipeline's metric instruments.

Logs go to stderr as JSON lines (``JsonFormatter``) and, when an OTLP endpoint is
configured, to the collector through ``OtlpLogShipper``. The shipper is a small
custom OTLP/HTTP exporter that runs on its own thread, never retries, and counts
every record it fails to deliver in ``pipeline.log.dropped``.

``Telemetry.shutdown`` finishes within ``_SHUTDOWN_BUDGET`` (10) seconds, even when
the collector hangs, leaving most of the platform's stop grace period for the rest of
the worker.

Public API:
    Env: Deployment environment names.
    TaskAttributes: The bound task's ``pipeline.*`` fields.
    Instruments: The pipeline's metric instruments.
    bind_task: Bind a task's ``pipeline.*`` fields for the duration of a ``with`` block.
    task_attributes: Return the bound task's ``pipeline.*`` fields.
    traced: Decorate a function so each call runs inside a span.
    ContextFilter: Logging filter that adds trace and task context to records.
    JsonFormatter: Logging formatter that writes one JSON line per record.
    OtlpLogShipper: Background OTLP/HTTP exporter for log records.
    Telemetry: Handles to the configured providers and handlers, with ``shutdown``.
    configure: Set up tracing, metrics and logging once per process.

"""

# STANDARD
import copy
import json
import logging
import logging.handlers
import os
import queue
import re
import sys
import threading
import time
import traceback
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
from urllib.parse import urlsplit

# 3rd PARTY
import httpx
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.psycopg import PsycopgInstrumentor
from opentelemetry.metrics import Counter, Histogram, UpDownCounter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode, Tracer
from opentelemetry.util.re import parse_env_headers

# LOCAL
from pipeline._wrapping import around

#: Deployment environment names.
type Env = Literal["local", "staging", "production"]
#: The bound task's ``pipeline.*`` fields.
type TaskAttributes = Mapping[str, str | int]

#: The bound task's fields; unset outside a task.
_task: ContextVar[TaskAttributes] = ContextVar("pipeline_task")  # the bound task's pipeline.* fields


@dataclass(frozen=True, slots=True)
class Instruments:
    """Hold the pipeline's OpenTelemetry metric instruments.

    ``configure`` builds one instance from the ``pipeline`` meter and hands it to the
    worker, the stages and the connector clients. Each attribute below gives the metric name,
    its unit, what one recording means and the attributes it carries.

    Attributes:
        task_duration: ``pipeline.task.duration`` (histogram, ``s``): wall-clock time of one
            task attempt, with ``job`` and ``outcome`` attributes. ``Lease`` records it once per
            committed outcome report, and the worker loop once per failure report for a claimed
            task no ``Lease`` could be built for; ``outcome`` is the ``Outcome`` value.
        rows: ``pipeline.rows`` (counter, ``{row}``): rows passing through a stage, with ``job``
            and ``stage`` attributes. ``stage`` is ``fetched`` (extract); ``waiting``,
            ``quarantined``, ``warned``, ``deduplicated`` or ``key_unreadable`` (a failed record
            whose key could not be read) (validate); ``written``, ``changed``, ``skipped`` (rows
            of a ``ReplacePartition`` table not written because records are held) or
            ``prune_skipped`` (the number of held records, added once for each table left
            unpruned), plus a ``table`` attribute (load).
        check_failures: ``pipeline.check.failures`` (counter, ``{failure}``): one per rule a
            record fails, with ``job``, ``rule`` and ``severity`` attributes.
        connector_requests: ``pipeline.connector.requests`` (counter, ``{request}``): one per
            HTTP request to a connector API, with ``connector``, ``status`` and ``result``
            attributes.
        text_replaced: ``pipeline.connector.text_replaced`` (counter, ``{response}``): one per
            response body whose NUL characters or lone surrogates (text Postgres cannot store)
            were replaced, with a ``connector`` attribute.
        log_dropped: ``pipeline.log.dropped`` (counter, ``{record}``): log records that never
            reached the collector, with a ``reason`` attribute of ``convert``,
            ``post_error``, ``http_status``, ``rejected``, ``shed``, ``shutdown`` or
            ``error``.
        slots_busy: ``pipeline.worker.slots_busy`` (up-down counter, ``{slot}``): task slots
            currently in use; incremented on claim, decremented on release.
        queue_wait: ``pipeline.task.queue_wait`` (histogram, ``s``): time a task waited in
            the queue before it was claimed.
        heartbeat_lag: ``pipeline.lease.heartbeat_lag`` (histogram, ``s``): time between a
            lease's successful heartbeats.
        lease_losses: ``pipeline.lease.losses`` (counter, ``{loss}``): one per lease lost,
            with ``kind`` and ``job`` attributes.
        lease_unwind: ``pipeline.lease.unwind`` (histogram, ``s``): time a task took to
            unwind after losing its lease, with ``kind`` and ``job`` attributes.
        unknown_jobs: ``pipeline.task.unknown_job`` (counter, ``{task}``): one per claimed
            task whose job is not registered.
        worker_paused: ``pipeline.worker.paused`` (counter, ``{pause}``): one each time the
            worker pauses claiming.

    """

    task_duration: Histogram  # pipeline.task.duration
    rows: Counter  # pipeline.rows
    check_failures: Counter  # pipeline.check.failures
    connector_requests: Counter  # pipeline.connector.requests
    text_replaced: Counter  # pipeline.connector.text_replaced
    log_dropped: Counter  # pipeline.log.dropped
    slots_busy: UpDownCounter  # pipeline.worker.slots_busy
    queue_wait: Histogram  # pipeline.task.queue_wait
    heartbeat_lag: Histogram  # pipeline.lease.heartbeat_lag
    lease_losses: Counter  # pipeline.lease.losses
    lease_unwind: Histogram  # pipeline.lease.unwind
    unknown_jobs: Counter  # pipeline.task.unknown_job
    worker_paused: Counter  # pipeline.worker.paused


@contextmanager
def bind_task(
    *, task_id: str, run_id: str, attempt: int, connector: str, job: str, source_id: str, partition: str
) -> Iterator[None]:
    """Bind a task's ``pipeline.*`` fields to the current context for the duration of a ``with`` block.

    The fields are stored in the ``_task`` context variable. ``ContextFilter`` copies them
    onto every log record and ``traced`` adds them to each span it starts. They are also set
    on the span that is current at entry, if it is recording. Spans created by the ``httpx``
    and ``psycopg`` instrumentation do not get them, but they share the task's trace.

    On exit the context variable is restored to its previous value, so log records and
    ``traced`` spans after an inner block carry the outer task's fields again. On the span
    current at entry, if it is an SDK span still recording at exit, each ``pipeline.*`` field
    it held before entry with a ``str`` or ``int`` value is set back to that value, so after an
    inner block it describes the outer task again. A field it did not hold before entry stays
    set: an OTel span cannot drop an attribute.

    Preconditions:
        None. Works before ``configure``; the current span is then non-recording, so only
        ``_task`` is set. The restore on exit assumes blocks that share an entry span nest. Two
        that overlap without nesting (for example concurrent asyncio tasks under one span) leave
        that span with one of their fields after both exit. ``_task`` is per-context and unaffected.

    Args:
        task_id: Stored as ``pipeline.task_id``.
        run_id: Stored as ``pipeline.run_id``.
        attempt: Stored as ``pipeline.attempt``.
        connector: Stored as ``pipeline.connector``. The external system the task fetches from, for
            example ``"tiktok_shop"``.
        job: Stored as ``pipeline.job``.
        source_id: Stored as ``pipeline.source_id``. The account the task is for.
        partition: Stored as ``pipeline.partition``.

    Yields:
        None.

    Raises:
        Nothing itself. An exception raised in the ``with`` block propagates unchanged, and
        ``_task`` is restored even when one does.

    """
    attrs: TaskAttributes = {
        "pipeline.task_id": task_id,
        "pipeline.run_id": run_id,
        "pipeline.attempt": attempt,
        "pipeline.connector": connector,
        "pipeline.job": job,
        "pipeline.source_id": source_id,
        "pipeline.partition": partition,
    }
    token = _task.set(attrs)
    span = trace.get_current_span()
    before: dict[str, str | int] = {}  # the entry span's fields this block overwrites
    try:
        if span.is_recording():
            held = span.attributes if isinstance(span, ReadableSpan) and span.attributes is not None else {}
            for key in attrs:
                if isinstance(value := held.get(key), str | int):  # pipeline.* fields are str or int
                    before[key] = value
            span.set_attributes(attrs)
        yield
    finally:
        if before and span.is_recording():
            span.set_attributes(before)
        _task.reset(token)


def task_attributes() -> TaskAttributes:  # the bound task's pipeline.* fields, empty outside a task
    """Return the bound task's ``pipeline.*`` fields, or an empty dict outside a task.

    Preconditions:
        None.

    Args:
        None.

    Returns:
        A new dict on every call. Mutating it does not change the bound task.

    Raises:
        Nothing.

    """
    return dict(_task.get({}))


def traced[F: Callable[..., Any]](name: str, **attrs: str) -> Callable[[F], F]:
    """Decorate a function so each call runs inside a span called ``name``.

    The span carries the current task's ``pipeline.*`` fields (see ``bind_task``) plus
    ``attrs``. Values in ``attrs`` take precedence if a key appears in both. Task fields
    are read when the span starts; outside ``bind_task`` the span has only ``attrs``.

    If the function raises an ``Exception``, the span records it, gets ``ERROR``
    status with its error class, and gets a ``pipeline.error_class`` attribute taken from
    the exception's ``error_class`` (or ``"Unexpected"`` if it has none). The full
    message stays in the exception event. Any other ``BaseException`` (for example
    ``CancelledError``) gives ``ERROR`` status with the class name and is not recorded.
    The exception is then re-raised. For a coroutine function the span stays open across the
    ``await`` (``around`` gives it an async wrapper).

    Recording an ``Exception`` reads ``err.error_class`` with ``getattr`` and calls the SDK's
    ``Span.record_exception``, which calls ``str(err)``. Both are guarded: if reading
    ``error_class`` raises, the class is ``"Unexpected"``; if ``record_exception`` raises, the
    span gets no exception event. Either way the span ends with ``ERROR`` status and
    ``pipeline.error_class``, and the original exception propagates.

    Preconditions:
        None. What may be decorated is checked; see Raises.

    Args:
        name: Span name.
        **attrs: Static string attributes added to every span.

    Returns:
        A decorator. It returns a wrapper with the same signature as the decorated function.

    Raises:
        TypeError: At decoration, from ``around``, if the decorated object is not a plain or
            coroutine function or method, or is a generator or async generator function. At call
            time, from ``around``, if a plain function returns an awaitable, a generator or an
            async generator.
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
            Exception: Whatever the wrapped call raised, re-raised unchanged after the span gets
                ``ERROR`` status and ``pipeline.error_class`` and records it. An ``Exception`` from
                reading ``err.error_class`` (the class is then ``"Unexpected"``) or from
                ``record_exception`` (``str(err)``; the event is then skipped) is suppressed.
            BaseException: Any other ``BaseException`` (for example ``CancelledError`` or
                ``KeyboardInterrupt``), re-raised unchanged after the span gets ``ERROR``
                status with the class name. It is not recorded and gets no ``pipeline.error_class``.

        """
        tracer = trace.get_tracer("pipeline")  # the global provider, read at call time
        with tracer.start_as_current_span(
            name,
            attributes={**task_attributes(), **attrs},
            record_exception=False,  # recorded below, once
            set_status_on_exception=False,
        ) as active:
            try:
                yield
            except Exception as err:
                named = None
                with suppress(Exception):  # an error_class property that raises must not replace err
                    named = getattr(err, "error_class", None)
                error_class = named if isinstance(named, str) else "Unexpected"
                active.set_status(Status(StatusCode.ERROR, error_class))
                active.set_attribute("pipeline.error_class", error_class)
                with suppress(Exception):  # str(err) may raise; err must still propagate
                    active.record_exception(err)
                raise
            except BaseException as err:
                active.set_status(Status(StatusCode.ERROR, type(err).__name__))
                raise

    return around(span)


class ContextFilter(logging.Filter):
    """Add trace and task context to log records. Never drops a record.

    Sets these attributes on each record:
        trace_id: Current trace ID as 32 lowercase hex characters.
        span_id: Current span ID as 16 lowercase hex characters.
        pipeline_task: Dict of the current task's ``pipeline.*`` fields (see ``bind_task``),
            empty outside a task.

    ``trace_id`` and ``span_id`` are only present when there is a valid current span; outside
    one, any values the caller passed through ``extra`` are removed.

    Attach this to handlers, not loggers. Logger filters do not run on records
    propagated from child loggers.

    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Add ``trace_id``, ``span_id`` and ``pipeline_task`` to ``record`` and keep it.

        Inside a valid current span, ``trace_id`` and ``span_id`` are set from it, overwriting
        any values the caller passed through ``extra``. Outside one, such values are removed,
        so the record carries neither, as for any record logged without a span; only real span
        ids are ever written or shipped. ``pipeline_task`` is always set to the current task's
        ``pipeline.*`` fields.

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
        else:  # a caller's extra must not pass for a span's ids
            vars(record).pop("trace_id", None)
            vars(record).pop("span_id", None)
        record.pipeline_task = task_attributes()
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
        pipeline.*: The task fields from ``record.pipeline_task``, flattened into the top level.
        exc: Formatted traceback, present when ``record.exc_info`` holds an exception (or, without
            one, when ``record.exc_text`` is set). A record logged with ``exc_info=True`` outside an
            ``except`` block carries ``(None, None, None)`` and gets no ``exc``, as
            ``_exception_attributes`` sends no exception for it.
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
            One line of JSON with no newline. ``exc`` is written only when ``record.exc_info`` holds an
            exception, or else ``record.exc_text`` is set; ``(None, None, None)`` writes none.

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
        line.update(getattr(record, "pipeline_task", {}))
        if record.exc_info and record.exc_info[1] is not None:  # exc_info=True outside an except gives no exception
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
_SEVERITY = (  # (min levelno, severityNumber, severityText)
    (logging.CRITICAL, 21, "CRITICAL"),
    (logging.ERROR, 17, "ERROR"),
    (logging.WARNING, 13, "WARNING"),
    (logging.INFO, 9, "INFO"),
    (0, 5, "DEBUG"),
)
#: Maps ``repr(float)`` output for non-finite values to the spellings used in
#: OTLP/JSON. Standard JSON cannot represent these values.
_NON_FINITE = {"nan": "NaN", "inf": "Infinity", "-inf": "-Infinity"}
#: Lower bound of the int64 range an OTLP ``intValue`` can hold; ``_any_value`` sends other ints as strings.
_INT64_MIN = -(2**63)
#: Upper bound of the int64 range an OTLP ``intValue`` can hold; ``_any_value`` sends other ints as strings.
_INT64_MAX = 2**63 - 1
#: Sentinel placed on the export queue to wake the worker thread. Compared by
#: identity, never exported.
_WAKE = logging.makeLogRecord({})  # put on the queue to wake the thread; never shipped
#: httpx timeout in seconds for each phase (connect, write, read, pool) of one export
#: POST. A slow POST can take longer than this in total.
_POST_TIMEOUT = 3.0  # seconds, httpx's limit for each phase of one post, and the exporters' limit for one request
#: Seconds ``Telemetry.shutdown`` allows for stopping the log shipper.
_SHIPPER_SHARE = 4.0  # seconds Telemetry.shutdown allows for stopping the log shipper
#: Seconds ``Telemetry.shutdown`` allows for shutting down the tracer provider.
_SPANS_SHARE = 3.0  # seconds it allows for shutting down the tracer provider
#: Seconds ``Telemetry.shutdown`` allows for shutting down the meter provider.
_METRICS_SHARE = 3.0  # seconds it allows for shutting down the meter provider
#: Seconds ``Telemetry.shutdown`` allows in total (10.0).
_SHUTDOWN_BUDGET = _SHIPPER_SHARE + _SPANS_SHARE + _METRICS_SHARE  # 10.0, the whole of shutdown
#: Backlog limit in batches. After each batch it ships, the export thread sheds the oldest
#: queued records until at most ``max_batch * _BACKLOG_BATCHES`` remain, counting them in
#: ``pipeline.log.dropped``. Between sheds (for example during a slow POST) the queue is
#: unbounded.
_BACKLOG_BATCHES = 10  # the thread sheds the oldest records beyond this many batches
#: An OTLP/JSON trace id: exactly 32 lowercase hex characters.
_TRACE_ID = re.compile(r"[0-9a-f]{32}")
#: An OTLP/JSON span id: exactly 16 lowercase hex characters.
_SPAN_ID = re.compile(r"[0-9a-f]{16}")
#: Longest ``exception.stacktrace`` sent, in characters; longer traces keep their end.
_MAX_STACKTRACE = 32_768  # characters of exception.stacktrace sent; a longer trace keeps its end


class _ShipperQueue(queue.SimpleQueue[logging.LogRecord]):
    """The shipper's queue: once closed by ``OtlpLogShipper.stop``, it counts each new record as dropped.

    While open, ``put`` and ``put_nowait`` enqueue as ``SimpleQueue`` does. Once ``close`` has run, they enqueue only
    ``_WAKE``; any other record is counted in ``dropped`` with reason ``shutdown`` instead, except a record from the
    shipper's own thread, which is discarded uncounted as the thread discards its own records. A lock makes each put
    land either before ``close`` (in the queue, which the export thread then ships or counts) or after it (counted).

    """

    def __init__(self, dropped: Counter) -> None:
        """Create an open, empty queue.

        Preconditions:
            None.

        Args:
            dropped: Counter incremented for every record refused after ``close``.

        Returns:
            None.

        Raises:
            Nothing.

        """
        self._dropped = dropped
        self._lock = threading.Lock()
        self._closed = False
        self._own: int | None = None  # the shipper thread's ident, whose records are never counted

    def put(self, item: logging.LogRecord, block: bool = True, timeout: float | None = None) -> None:
        """Enqueue ``item`` while the queue is open; once closed, count it as dropped instead.

        Preconditions:
            None.

        Args:
            item: The record to enqueue, or ``_WAKE``.
            block: Ignored, as by ``SimpleQueue.put``.
            timeout: Ignored, as by ``SimpleQueue.put``.

        Returns:
            None.

        Raises:
            Exception: Any exception from ``dropped``, unchanged.

        """
        with self._lock:
            if item is _WAKE or not self._closed:
                super().put(item)
                return
        if item.thread != self._own:
            self._dropped.add(1, {"reason": "shutdown"})

    def put_nowait(self, item: logging.LogRecord) -> None:
        """Enqueue ``item`` as ``put`` does; ``QueueHandler.enqueue`` calls this.

        Preconditions:
            None.

        Args:
            item: The record to enqueue.

        Returns:
            None.

        Raises:
            Exception: Any exception from ``dropped``, unchanged.

        """
        self.put(item)

    def close(self, own: int | None) -> None:
        """Refuse every later record but ``_WAKE``, counting it as dropped unless it comes from thread ``own``.

        Preconditions:
            None.

        Args:
            own: Ident of the shipper thread, or ``None``.

        Returns:
            None.

        Raises:
            Nothing.

        """
        with self._lock:
            self._closed = True
            self._own = own

    def reopen(self) -> None:
        """Accept records again, as before ``close``.

        Preconditions:
            None.

        Args:
            None.

        Returns:
            None.

        Raises:
            Nothing.

        """
        with self._lock:
            self._closed = False


class OtlpLogShipper:
    """Batch log records from a queue and POST them to an OTLP/HTTP collector.

    ``_EagerQueueHandler`` puts records on ``queue`` with ``msg`` already formatted and any
    exception stored in ``otel_exception``. A background daemon thread takes them off in
    batches of up to ``max_batch`` records, waiting at most ``max_wait`` seconds after the
    first record of a batch, and POSTs each batch to ``{endpoint}/v1/logs`` as an OTLP JSON
    export request. Records are grouped into one scope per logger name.

    Failed batches are not retried. Every record that is lost to a failed POST, a
    conversion error, a collector rejection, a full backlog, shutdown or an unexpected
    error is added to ``dropped``, with a ``reason`` attribute. Once ``stop`` has begun, the queue
    (``_ShipperQueue``) counts every further record as dropped with reason ``shutdown`` instead of enqueuing it.

    Records logged on the shipper's own thread (for example by ``httpx``) are
    discarded to prevent a feedback loop. They are not counted as dropped.

    """

    def __init__(
        self,
        endpoint: str,
        resource: Resource,
        dropped: Counter,
        *,
        headers: Mapping[str, str] | None = None,
        max_batch: int = 500,
        max_wait: float = 2.0,
    ) -> None:
        """Create a shipper. Call ``start`` to begin exporting.

        Preconditions:
            ``endpoint`` is a base URL without ``/v1/logs``.

        Args:
            endpoint: Collector base URL, without ``/v1/logs``.
            resource: OTel resource whose attributes are sent with every batch.
            dropped: Counter incremented for every record that is not delivered.
            headers: Headers sent with every POST, for example an
                ``authorization`` header the collector requires. Defaults to none.
            max_batch: Maximum records per POST. Must be at least 1. Also sets the
                backlog limit to ``max_batch * _BACKLOG_BATCHES`` records.
            max_wait: Maximum seconds to wait for a batch to fill after its first
                record arrives. Must be greater than 0 and at most ``threading.TIMEOUT_MAX``
                (about 9.2e9 seconds), the largest timeout ``queue.get`` in ``_collect`` accepts.

        Returns:
            None.

        Raises:
            ValueError: If ``max_batch < 1``, or ``max_wait`` is not greater than 0 or is
                greater than ``threading.TIMEOUT_MAX`` (including NaN and ``inf``).

        """
        if max_batch < 1 or not 0 < max_wait <= threading.TIMEOUT_MAX:
            raise ValueError("max_batch must be >= 1 and max_wait > 0 and <= threading.TIMEOUT_MAX")
        self._url = endpoint.rstrip("/") + "/v1/logs"
        self._resource = _attributes(resource.attributes)
        self._dropped = dropped
        self._headers = dict(headers or {})
        self._max_batch = max_batch
        self._max_wait = max_wait
        self._queue = _ShipperQueue(dropped)
        self._transport: httpx.BaseTransport | None = None  # test hook
        self._stopping = threading.Event()
        self._deadline = 0.0
        self._thread: threading.Thread | None = None

    @property
    def queue(self) -> queue.SimpleQueue[logging.LogRecord]:
        """Return the queue to pass to the ``QueueHandler`` that feeds this shipper.

        Preconditions:
            None.

        Args:
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

        Args:
            None.

        Returns:
            None.

        Raises:
            OSError, ssl.SSLError: From ``httpx.Client`` if TLS setup fails (for example the CA
                bundle cannot be loaded).
            UnicodeEncodeError: From ``httpx.Client``, which encodes header names and values as
                ASCII, if a name or value in ``headers`` is not ASCII (``configure`` drops such a
                header before building the shipper). No thread is started.
            RuntimeError: If a previous thread is still alive after a ``stop`` whose join timed
                out, so two export threads never share the queue. Also if the OS cannot create the
                thread: the HTTP client is then closed and the shipper stays stopped, so ``start``
                can be retried.

        """
        if self._thread is not None and self._thread.is_alive():
            if self._stopping.is_set():
                raise RuntimeError("OtlpLogShipper cannot start: the previous export thread is still stopping")
            return
        self._stopping.clear()
        self._queue.reopen()
        client = httpx.Client(timeout=httpx.Timeout(_POST_TIMEOUT), headers=self._headers, transport=self._transport)
        HTTPXClientInstrumentor.uninstrument_client(client)  # the shipper's own posts must make no spans
        thread = threading.Thread(target=self._run, args=(client,), name="otlp-log-shipper", daemon=True)
        try:
            thread.start()
        except BaseException:
            client.close()
            raise
        self._thread = thread

    def stop(self, timeout: float) -> None:
        """Export queued records, then stop the background thread.

        First closes the queue, so every record enqueued from then on, a log call already in flight on another
        thread included, is counted as dropped with reason ``shutdown`` instead (see ``_ShipperQueue``). Then waits up
        to ``timeout`` seconds. The thread first ships the batch it is collecting,
        then ships further batches only while more than ``_POST_TIMEOUT`` seconds remain before
        the deadline, so with ``timeout`` under ``_POST_TIMEOUT`` nothing else queued is sent.
        Every POST started after this call gives each httpx phase a quarter of the time left
        before the deadline, at most ``_POST_TIMEOUT`` (see ``_post_timeout``). httpx applies
        the read and write limits to each socket operation, so a collector that keeps trickling
        bytes can still hold a POST past the deadline; ``join`` returns at ``timeout`` regardless.
        A POST already in flight when this is called keeps its full ``_POST_TIMEOUT``. What is
        left is discarded and counted as dropped with reason ``shutdown`` when the thread finishes.
        Does nothing if not started.

        Preconditions:
            Called from one thread, not concurrently with ``start``.

        Args:
            timeout: Seconds to wait for queued records to be sent and the thread to
                finish; at least 0 and at most ``threading.TIMEOUT_MAX`` (about 9.2e9 seconds),
                the largest timeout ``join`` accepts. ``Telemetry.shutdown`` passes at most
                ``_SHIPPER_SHARE``.

        Returns:
            None. If ``join`` times out it returns with the thread still running and kept in
            ``_thread``, so ``start`` refuses to start a second one, after logging a WARNING
            through this module's logger.

        Raises:
            ValueError: If ``timeout`` is below 0 or greater than ``threading.TIMEOUT_MAX``
                (including NaN and ``inf``), checked first, so the thread is not told to stop.

        """
        if not 0 <= timeout <= threading.TIMEOUT_MAX:
            raise ValueError("timeout must be >= 0 and <= threading.TIMEOUT_MAX")
        thread = self._thread
        if thread is None:
            return
        self._queue.close(thread.ident)
        self._deadline = time.monotonic() + timeout
        self._stopping.set()
        self._queue.put(_WAKE)
        thread.join(timeout=timeout)
        if thread.is_alive():
            logging.getLogger(__name__).warning(
                "log shipper did not stop within %.1fs; records still queued are dropped", timeout
            )
            return
        self._thread = None

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
                except Exception as err:  # the thread must survive; count what it held, say one line
                    print(f"otlp-log-shipper: {type(err).__name__}: {err}", file=sys.stderr)
                    self._dropped.add(len(batch), {"reason": "error"})
                    self._stopping.wait(0.1)
            while time.monotonic() < self._deadline - _POST_TIMEOUT:
                rest: list[logging.LogRecord] = []
                try:
                    self._take(own, rest)
                    if not rest:
                        break
                    self._ship(client, rest)
                except Exception as err:
                    print(f"otlp-log-shipper: {type(err).__name__}: {err}", file=sys.stderr)
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
        while True:  # no timeout: idle until something shippable arrives
            first = self._queue.get()
            if first is _WAKE:
                return
            if first.thread != own:
                break
        batch.append(first)
        end = time.monotonic() + self._max_wait  # the clock starts at the first shippable record
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
        every converted record is counted as dropped. On a 2xx response, the count of records
        the collector reports as rejected in ``partialSuccess`` is clamped to between 0 and
        the number of converted records, then counted (``rejected``).
        Every string in the body but ``traceId`` and ``spanId`` has its lone surrogates escaped
        (``_text``); those two are copied only when they are lowercase hex ids of the OTLP
        length (see ``_log_record``), so they never need escaping. The POST's per-phase timeout
        comes from ``_post_timeout``.

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
        scopes: dict[str, list[dict[str, Any]]] = {}  # dicts keep first-seen order
        built = 0
        for record in batch:
            try:
                scopes.setdefault(record.name, []).append(_log_record(record))
                built += 1
            except Exception:  # one bad record is dropped, not the batch
                self._dropped.add(1, {"reason": "convert"})
        if not built:
            return
        try:
            response = client.post(
                self._url, json=_export_request(self._resource, scopes), timeout=self._post_timeout()
            )
        except Exception:  # httpx errors and anything else: count, never raise
            self._dropped.add(built, {"reason": "post_error"})
        else:
            if not response.is_success:
                self._dropped.add(built, {"reason": "http_status"})
            else:
                rejected = min(max(_rejected_count(response), 0), built)
                if rejected:
                    self._dropped.add(rejected, {"reason": "rejected"})

    def _post_timeout(self) -> float:
        """Return the httpx per-phase timeout, in seconds, for a POST starting now.

        Preconditions:
            Runs on the shipper thread only. ``stop`` sets ``_deadline`` before ``_stopping``.

        Args:
            None.

        Returns:
            ``_POST_TIMEOUT`` before ``stop`` is called. Once it has been, a quarter of the
            seconds left before ``_deadline`` (0.0 once it has passed), at most
            ``_POST_TIMEOUT``. The four phases' limits then sum to no more than the time left,
            though httpx applies the read and write limits per socket operation, not per POST.

        Raises:
            Nothing.

        """
        if not self._stopping.is_set():
            return _POST_TIMEOUT
        return min(_POST_TIMEOUT, max(0.0, self._deadline - time.monotonic()) / 4)


def _text(value: str) -> str:
    """Return ``value`` with each lone surrogate replaced by its backslash escape, as plain text.

    httpx encodes a JSON body as UTF-8, which fails on a lone surrogate (for example from
    ``os.fsdecode`` of a non-UTF-8 file name), losing the whole batch.

    Preconditions:
        None.

    Args:
        value: Text to send over OTLP.

    Returns:
        ``value`` unchanged when it encodes as UTF-8, otherwise a copy with every lone
        surrogate replaced by its ``backslashreplace`` escape (six ASCII characters).

    Raises:
        Nothing.

    """
    return value.encode("utf-8", "backslashreplace").decode("utf-8")


def _rejected_count(response: httpx.Response) -> int:
    """Return how many log records the collector rejected in a 2xx export response.

    Reads ``partialSuccess.rejectedLogRecords`` from the OTLP/JSON response body. The
    count may be a string, as OTLP/JSON encodes int64 values that way.

    Preconditions:
        ``response`` is a 2xx response to an OTLP/HTTP logs export, with its body read.

    Args:
        response: The collector's response.

    Returns:
        ``int()`` of the rejected count, or 0 when the body is empty, not JSON, not an object,
        has no ``partialSuccess`` or no ``rejectedLogRecords``, or ``int()`` fails on the
        count. A JSON number such as ``2.5`` is truncated to 2. The count is not bounded: it
        may be negative or larger than the batch that was sent.

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
    ``scopes``. The scope name is the key, which ``_ship`` sets to the logger name, with
    lone surrogates escaped by ``_text``.

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
                "scopeLogs": [
                    {"scope": {"name": _text(name)}, "logRecords": records} for name, records in scopes.items()
                ],
            }
        ]
    }


def _log_record(record: logging.LogRecord) -> dict[str, Any]:
    """Convert a log record to an OTLP/JSON ``LogRecord``.

    Sets:
        timeUnixNano: ``record.created`` in nanoseconds, as a string.
        severityNumber, severityText: From the first ``_SEVERITY`` row whose
            minimum level ``record.levelno`` meets.
        body: ``str(record.msg)``, already formatted by ``_EagerQueueHandler``, with lone
            surrogates escaped by ``_text``.
        traceId, spanId: ``record.trace_id`` and ``record.span_id`` copied unchanged when both
            are ``str`` and match ``_TRACE_ID`` (32 lowercase hex characters) and ``_SPAN_ID``
            (16) in full; otherwise both are left out of the record. ``ContextFilter`` writes
            them as hex ids, but a record that reached the queue without it may carry any text
            from ``extra``.
        attributes: The ``pipeline_task`` fields merged with ``otel_exception``, with
            exception fields taking precedence on a key collision, plus
            ``code.stacktrace`` from ``record.stack_info`` when it is set. Omitted when
            all are empty.

    Preconditions:
        ``record`` was prepared by ``_EagerQueueHandler`` with ``ContextFilter`` on the same handler.

    Args:
        record: The prepared record. Not modified.

    Returns:
        A new dict.

    Raises:
        StopIteration: If ``record.levelno`` is negative, since no ``_SEVERITY``
            row matches. ``_ship`` catches this and counts the record as dropped.
        Exception: Whatever ``_attributes`` raises for a ``pipeline_task`` or
            ``otel_exception`` value. ``_ship`` catches it and counts the record as dropped.

    """
    number, text = next((num, name) for floor, num, name in _SEVERITY if record.levelno >= floor)
    out: dict[str, Any] = {
        "timeUnixNano": str(round(record.created * 1_000_000_000)),
        "severityNumber": number,
        "severityText": text,
        "body": {"stringValue": _text(str(record.msg))},  # already formatted by the queue handler
    }
    trace_id = getattr(record, "trace_id", None)
    span_id = getattr(record, "span_id", None)
    if (
        isinstance(trace_id, str)
        and isinstance(span_id, str)
        and _TRACE_ID.fullmatch(trace_id)
        and _SPAN_ID.fullmatch(span_id)
    ):
        out["traceId"] = trace_id
        out["spanId"] = span_id
    task = getattr(record, "pipeline_task", None)
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
        exception.message: ``str(err)``, or ``<unprintable {type}>`` if that raises.
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
        Nothing: ``str(err)`` is guarded, and ``traceback.format_exception`` reports an
        unprintable exception instead of raising.

    """
    err = record.exc_info[1] if record.exc_info else None
    if err is None:
        return {}
    cls = type(err)
    name = cls.__qualname__ if cls.__module__ == "builtins" else f"{cls.__module__}.{cls.__qualname__}"
    try:
        message = str(err)
    except Exception:  # an unprintable exception must not lose the record
        message = f"<unprintable {name}>"
    stack = "".join(traceback.format_exception(err))
    if len(stack) > _MAX_STACKTRACE:
        stack = f"[truncated {len(stack) - _MAX_STACKTRACE} chars]\n" + stack[-_MAX_STACKTRACE:]
    return {
        "exception.type": name,
        "exception.message": message,
        "exception.stacktrace": stack,
    }


def _attributes(values: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Convert a mapping to a list of OTLP/JSON ``KeyValue`` entries.

    Each key has its lone surrogates escaped by ``_text``, and each value is converted with
    ``_any_value``. Order follows the mapping.

    Preconditions:
        None.

    Args:
        values: Attribute names and values.

    Returns:
        A new list with one ``KeyValue`` dict per item.

    Raises:
        Exception: Whatever ``_any_value`` raises for a value.

    """
    return [{"key": _text(key), "value": _any_value(value)} for key, value in values.items()]


def _any_value(value: object) -> dict[str, Any]:
    """Convert a Python value to an OTLP/JSON ``AnyValue``.

    Conversions:
        bool: ``boolValue``. Checked before ``int``, since ``bool`` subclasses it.
        int: ``intValue``, as a decimal string as OTLP/JSON requires for int64, when it is
            within the int64 range (``-2**63`` to ``2**63 - 1``); outside it, ``stringValue``
            holding the same decimal string, since the collector cannot parse it as an int64.
        float: ``doubleValue``. NaN and infinities become the strings ``"NaN"``,
            ``"Infinity"`` and ``"-Infinity"`` via ``_NON_FINITE``.
        list, tuple: ``arrayValue``, converting each item recursively.
        Anything else: ``stringValue`` using ``str(value)``, with lone surrogates escaped
            by ``_text``.

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
    if isinstance(value, bool):  # before int: bool is an int subclass
        return {"boolValue": value}
    if isinstance(value, int):
        if _INT64_MIN <= value <= _INT64_MAX:
            return {"intValue": str(value)}  # OTLP JSON int64 is a decimal string
        return {"stringValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": _NON_FINITE.get(repr(value), value)}
    if isinstance(value, list | tuple):
        return {"arrayValue": {"values": [_any_value(item) for item in value]}}
    return {"stringValue": _text(str(value))}


@dataclass(frozen=True, slots=True)
class Telemetry:
    """Handles to the tracing, metrics and logging set up for the pipeline.

    Call ``shutdown`` once at process exit to flush and detach everything.

    Attributes:
        tracer: Tracer for pipeline spans.
        instruments: Metric instruments (see ``Instruments``).
        tracer_provider: Provider that owns span export.
        meter_provider: Provider that owns metric export.
        log_shipper: Ships logs to the collector. ``configure`` sets it to ``None`` when no
            collector is configured, for example in development.
        queue_handler: Root logger handler that feeds ``log_shipper``. ``configure`` sets it
            to ``None`` exactly when ``log_shipper`` is ``None``; nothing else enforces that.
            ``shutdown`` detaches it whenever it is set.
        stderr_handler: Root logger handler that writes JSON lines to stderr. ``configure``
            always sets and attaches it; ``shutdown`` detaches it whenever it is set.

    """

    tracer: Tracer
    instruments: Instruments
    tracer_provider: TracerProvider
    meter_provider: MeterProvider
    log_shipper: OtlpLogShipper | None  # None when running locally without a collector
    queue_handler: logging.Handler | None = None  # on the root logger; feeds the shipper
    stderr_handler: logging.Handler | None = None  # on the root logger; JSON lines
    _once: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False, compare=False)

    def shutdown(self) -> None:
        """Flush and shut down all telemetry within ``_SHUTDOWN_BUDGET`` seconds. Only the first call has any effect.

        In order:
            1. Detach ``queue_handler`` from the root logger, so no new records
               reach the shipper.
            2. Stop ``log_shipper``, exporting queued records, allowing at most
               ``_SHIPPER_SHARE`` seconds. A record a log call already in flight
               still hands the shipper is counted as dropped (see ``OtlpLogShipper.stop``).
            3. Remove the ``httpx`` and ``psycopg`` instrumentation process-wide.
            4. Shut down the tracer provider, exporting pending spans, allowing at
               most ``_SPANS_SHARE`` seconds.
            5. Shut down the meter provider, exporting final metric values,
               including drops counted while stopping the shipper if its thread
               finished within its share, allowing at most ``_METRICS_SHARE`` seconds.
            6. Detach ``stderr_handler``.

        All steps share one deadline, ``_SHUTDOWN_BUDGET`` seconds after the call
        starts. Each slow step gets the smaller of its share and the time left, so
        time a step does not use is not passed on to later steps. A provider
        shutdown that overruns is abandoned on a daemon thread and may still be
        running when this returns.

        Each step runs through ``_attempt``, so a failure in one does not stop the
        rest. Logs produced during steps 2 to 5 go to stderr only.

        Preconditions:
            None. Safe to call more than once; later calls return at once.

        Args:
            None.

        Returns:
            None.

        Raises:
            Nothing for an ``Exception`` raised by a step: each step runs in ``_attempt``, which
            logs it and continues.
            BaseException: For example ``KeyboardInterrupt``, propagates from the step it
                interrupts and skips the remaining steps. Later calls do not retry them.
                A ``BaseException`` raised inside a provider shutdown ends that step's
                worker thread only and does not propagate.

        """
        if not self._once.acquire(blocking=False):
            return
        deadline = time.monotonic() + _SHUTDOWN_BUDGET

        def remaining() -> float:
            """Return the seconds left before the shutdown deadline.

            Preconditions:
                None.

            Args:
                None.

            Returns:
                Seconds until ``deadline``, or ``0.0`` once it has passed.

            Raises:
                Nothing.

            """
            return max(0.0, deadline - time.monotonic())

        root = logging.getLogger()
        if (queue_handler := self.queue_handler) is not None:
            _attempt("detach queue handler", lambda: root.removeHandler(queue_handler))
        if (shipper := self.log_shipper) is not None:
            _attempt("stop log shipper", lambda: shipper.stop(min(_SHIPPER_SHARE, remaining())))
        _attempt("uninstrument httpx", HTTPXClientInstrumentor().uninstrument)
        _attempt("uninstrument psycopg", PsycopgInstrumentor().uninstrument)
        _attempt("shut down tracer provider", self.tracer_provider.shutdown, timeout=min(_SPANS_SHARE, remaining()))
        _attempt("shut down meter provider", self.meter_provider.shutdown, timeout=min(_METRICS_SHARE, remaining()))
        if (stderr_handler := self.stderr_handler) is not None:
            _attempt("detach stderr handler", lambda: root.removeHandler(stderr_handler))


def _attempt(step: str, action: Callable[[], object], timeout: float | None = None) -> None:
    """Run one shutdown step, logging any exception instead of raising it.

    Used by ``Telemetry.shutdown`` so that one failing step does not prevent the
    rest from running. The return value of ``action`` is ignored.

    With ``timeout``, ``action`` runs on a daemon thread named
    ``"telemetry-shutdown: {step}"`` and is waited for at most ``timeout`` seconds.
    If it has not finished by then, a WARNING is logged and the thread is
    abandoned: it keeps running, and does not keep the process alive at exit.

    If the OS cannot start that thread (for example a container at its pids limit), the
    ``RuntimeError`` is logged at ERROR like a failing ``action`` and ``action`` does not run.

    Preconditions:
        ``timeout``, if given, is at least 0.

    Args:
        step: Short description used in the log messages and the thread name.
        action: Zero-argument callable to run.
        timeout: Seconds to wait for ``action``, or ``None`` to run it on the
            calling thread with no limit.

    Returns:
        None, once ``action`` has finished or, with ``timeout``, once the wait has
        run out.

    Raises:
        Nothing for ``Exception``: it is caught and logged at ERROR with its traceback.
        BaseException: Propagates from ``action`` when ``timeout`` is ``None``. With
            ``timeout``, it ends the worker thread and is reported by
            ``threading.excepthook`` instead.

    Notes:
        The failure is logged through ``logging``, so where it ends up depends on
        which handlers are still attached at that step: a failure in step 1 still
        reaches the shipper and stderr, one in steps 2 to 5 reaches stderr only, and a
        WARNING or ERROR after ``stderr_handler`` is detached goes to ``logging.lastResort``.

    """

    def run() -> None:
        """Run ``action``, logging any exception instead of raising it.

        Preconditions:
            None.

        Args:
            None.

        Returns:
            None.

        Raises:
            Nothing for ``Exception``: it is caught and logged at ERROR with its traceback.
            BaseException: Propagates from ``action``.

        """
        try:
            action()
        except Exception:
            logging.getLogger(__name__).exception("telemetry shutdown: %s failed", step)

    if timeout is None:
        run()
        return
    worker = threading.Thread(target=run, name=f"telemetry-shutdown: {step}", daemon=True)
    try:
        worker.start()
    except RuntimeError:  # the OS cannot create another thread
        logging.getLogger(__name__).exception("telemetry shutdown: %s failed", step)
        return
    worker.join(timeout=timeout)
    if worker.is_alive():
        logging.getLogger(__name__).warning(
            "telemetry shutdown: %s did not finish within %.1fs; abandoned", step, timeout
        )


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
                or ``"<unprintable message: {ExcType}>"`` (the class name of the exception that
                fallback raised) if that fails too.
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
            Nothing: every formatting failure is caught, and ``_exception_attributes`` does not
            raise.

        """
        out = copy.copy(record)  # the stderr handler keeps the original, with its exc_info
        try:
            out.msg = record.getMessage()
        except Exception:  # bad format args must not lose the record
            try:
                out.msg = f"{record.msg} (args: {record.args!r})"
            except Exception as err:  # an unprintable message still ships
                out.msg = f"<unprintable message: {type(err).__name__}>"
        out.args = None
        out.otel_exception = _exception_attributes(record)  # dict[str, str]
        out.exc_info = None  # no traceback frames stay alive in the queue
        out.exc_text = None
        return out


#: An HTTP header name: one or more token characters (RFC 9110 ``tchar``).
_HEADER_NAME = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")
#: An HTTP header value: visible ASCII characters with spaces and tabs between them (RFC 9110 ``field-value``
#: without ``obs-text``, which httpx's ASCII encoding refuses), or nothing.
_HEADER_VALUE = re.compile(r"(?:[\x21-\x7e](?:[\x20\x09\x21-\x7e]*[\x21-\x7e])?)?")


def _valid_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Return the headers whose name and value HTTP allows, logging a warning for each one dropped.

    The warning names the header (``repr`` of its name) and never its value, which may be a credential.

    Preconditions:
        None.

    Args:
        headers: Header names and values, as ``parse_env_headers`` returns them (URL-decoded).

    Returns:
        A new dict of the pairs whose name fully matches ``_HEADER_NAME`` and whose value fully matches
        ``_HEADER_VALUE``, in their original order.

    Raises:
        Nothing.

    """
    valid: dict[str, str] = {}
    for name, value in headers.items():
        if _HEADER_NAME.fullmatch(name) and _HEADER_VALUE.fullmatch(value):
            valid[name] = value
        else:
            logging.getLogger(__name__).warning("OTLP log header %r dropped: its name or value is not valid HTTP", name)
    return valid


def _is_http_url(endpoint: str) -> bool:
    """Return whether ``endpoint`` is an ``http://`` or ``https://`` URL with a host.

    Preconditions:
        None.

    Args:
        endpoint: The endpoint to check.

    Returns:
        True when ``urlsplit`` gives the scheme ``http`` or ``https`` (in any case) and a non-empty host;
        False otherwise, including when ``urlsplit`` raises ``ValueError`` (for example an invalid IPv6 host).

    Raises:
        Nothing.

    """
    try:
        parts = urlsplit(endpoint)
        return parts.scheme.lower() in {"http", "https"} and bool(parts.hostname)
    except ValueError:
        return False


def _positive_env[T: (int, float)](name: str, parse: Callable[[str], T], default: T, maximum: float) -> T:
    """Return the environment variable ``name`` parsed by ``parse``, or ``default`` when it is unset or unusable.

    A value that ``parse`` refuses with ``ValueError``, or that is not above zero (``NaN`` included), or that is
    above ``maximum`` (``inf`` included, from ``"inf"`` or ``"1e400"`` when ``parse`` is ``float``), is logged at
    WARNING, naming the variable and the default but never the value, and ``default`` is returned instead.

    Preconditions:
        ``default`` is above zero and at most ``maximum``.

    Args:
        name: The environment variable to read.
        parse: ``int`` or ``float``, as the SDK parses the variable.
        default: The SDK's default for the variable.
        maximum: The largest value the SDK accepts for the variable (``_MAX_SDK_MILLIS`` for a delay or interval,
            ``sys.maxsize`` for a size).

    Returns:
        The parsed value when the variable is set, parses, is above zero and is at most ``maximum``; ``default``
        otherwise.

    Raises:
        Nothing.

    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = parse(raw)
    except ValueError:
        value = None
    if value is not None and value > maximum:  # inf included; the SDK would overflow on it
        logging.getLogger(__name__).warning("%s is above %s; using the default %s", name, maximum, default)
        return default
    if value is not None and value > 0:
        return value
    logging.getLogger(__name__).warning("%s is not a positive number; using the default %s", name, default)
    return default


#: The longest delay or interval, in milliseconds, the SDK's export threads can wait for: ``threading.TIMEOUT_MAX``
#: seconds. A longer one makes their first ``Event.wait`` raise ``OverflowError``.
_MAX_SDK_MILLIS = threading.TIMEOUT_MAX * 1e3

#: Serialises ``configure``.
_lock = threading.Lock()
#: The ``Telemetry`` of the first ``configure`` that reached a global side effect; set once,
#: whether that call then succeeded or failed.
_active: Telemetry | None = None


def configure(*, env: Env, worker_id: str, log_level: str) -> Telemetry:
    """Set up tracing, metrics and logging for this process and return the handles.

    Creates a tracer provider and meter provider, registers them as the global
    providers, instruments ``httpx`` and ``psycopg``, and attaches handlers to the
    root logger:
        - A JSON stderr handler, always.
        - A queue handler feeding an ``OtlpLogShipper``, when an OTLP endpoint is set.

    The OTLP endpoint comes from ``OTEL_EXPORTER_OTLP_ENDPOINT``. It is required
    unless ``env`` is ``"local"``, and must be an ``http://`` or ``https://`` URL with a host
    (``_is_http_url``); in ``"local"`` a value that is not is logged at WARNING and treated as
    unset. Without it, spans and metrics are created but not exported, and logs go to stderr
    only. The span and metric exporters each allow ``_POST_TIMEOUT`` seconds per export
    request, so that shutdown can stay within ``_SHUTDOWN_BUDGET``.

    All three signals use one cleaned base: ``OTEL_EXPORTER_OTLP_ENDPOINT`` with surrounding whitespace and
    trailing slashes removed. The span exporter is given ``{base}/v1/traces``, the metric exporter
    ``{base}/v1/metrics`` and the log shipper posts to ``{base}/v1/logs``. Only
    ``OTEL_EXPORTER_OTLP_ENDPOINT`` is read: the per-signal ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT``,
    ``OTEL_EXPORTER_OTLP_METRICS_ENDPOINT`` and ``OTEL_EXPORTER_OTLP_LOGS_ENDPOINT`` are ignored, since an
    explicit endpoint overrides them in the exporters, so the signals cannot be split.

    The batch span processor's ``OTEL_BSP_MAX_QUEUE_SIZE``, ``OTEL_BSP_SCHEDULE_DELAY`` and
    ``OTEL_BSP_MAX_EXPORT_BATCH_SIZE`` and the metric reader's ``OTEL_METRIC_EXPORT_INTERVAL`` are read
    here (``_positive_env``) and passed to the SDK explicitly: a value that is malformed, not
    positive, or too large for the SDK (a delay or interval above ``threading.TIMEOUT_MAX`` seconds,
    ``inf`` included, or a size above ``sys.maxsize``) is logged at WARNING, naming the variable, and
    replaced by the SDK's default, so building the processor and reader does not raise for it and no
    export thread dies on it. A batch size, given or default, above the queue size is clamped to the
    queue size with a WARNING naming both variables.

    The log shipper sends the headers in ``OTEL_EXPORTER_OTLP_LOGS_HEADERS``, or, when that
    is unset or empty, in ``OTEL_EXPORTER_OTLP_HEADERS``, parsed as the OpenTelemetry SDK
    parses them (comma-separated ``key=value`` pairs, URL-decoded; malformed pairs skipped).
    A pair whose decoded name is not an HTTP token or whose decoded value is not visible ASCII
    with inner spaces or tabs is dropped with a warning naming the header, never its value
    (``_valid_headers``), before anything is registered.
    The span and metric exporters read their own headers from the environment.

    Resource attributes, merged over those in ``OTEL_RESOURCE_ATTRIBUTES`` (which is where
    ``service.version`` comes from, if it is set):
        service.name: ``OTEL_SERVICE_NAME``, or ``pipeline-worker``.
        deployment.environment.name: ``env``.
        service.instance.id: ``worker_id``. Must be unique per running worker.

    The root logger is set to ``log_level``. The ``httpx`` and ``httpcore`` loggers
    are set to WARNING to suppress per-request logs.

    Can only be called once per process. Call ``Telemetry.shutdown`` before exit. Nothing is
    registered with ``atexit``: the ``TracerProvider`` and ``MeterProvider`` are built with
    ``shutdown_on_exit=False``, so ``Telemetry.shutdown`` is the only flush and stays within
    ``_SHUTDOWN_BUDGET``. A process that exits without it does not flush spans or metrics.

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
        RuntimeError: If an earlier call reached registering the providers, whether it then
            succeeded or failed, or if ``OTEL_EXPORTER_OTLP_ENDPOINT`` is missing or not an
            ``http://`` or ``https://`` URL when ``env`` is not ``"local"``. A call after one that
            failed before registering the providers (an unknown level or a missing or bad endpoint)
            does not raise for that reason.
        ValueError: If ``log_level`` is not a known level name. Malformed, non-positive and too large ``OTEL_*``
            values are not raised: they are logged and replaced as described above. Should the SDK
            still refuse a processor or reader, its error is re-raised after the tracer provider is
            shut down, so no export thread is left running. A log header that HTTP does not allow
            (for example ``OTEL_EXPORTER_OTLP_HEADERS="x-team=%C3%A9"``) is logged and dropped, so
            ``OtlpLogShipper.start`` is not given one.
        BaseException: Anything raised while registering providers, instrumenting, starting the
            log shipper or attaching handlers is re-raised after ``Telemetry.shutdown`` has
            cleaned up; later calls then raise ``RuntimeError``.

    """
    global _active
    with _lock:
        if _active is not None:
            raise RuntimeError("telemetry is already configured; configure() runs once per process")
        level = logging.getLevelNamesMapping().get(log_level.upper())
        if level is None:
            raise ValueError(f"unknown log level {log_level!r}")
        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip().rstrip("/")
        if endpoint and not _is_http_url(endpoint):
            if env != "local":  # never the value: it may carry credentials
                raise RuntimeError(
                    f"OTEL_EXPORTER_OTLP_ENDPOINT must be an http:// or https:// URL when env is {env!r}"
                )
            logging.getLogger(__name__).warning("OTEL_EXPORTER_OTLP_ENDPOINT is not an http:// or https:// URL; unset")
            endpoint = ""
        if not endpoint and env != "local":
            raise RuntimeError(f"OTEL_EXPORTER_OTLP_ENDPOINT is required when env is {env!r}")

        resource = Resource.create(
            {
                "service.name": os.environ.get("OTEL_SERVICE_NAME") or "pipeline-worker",
                "deployment.environment.name": env,
                "service.instance.id": worker_id,
            }
        )
        span_exporter = OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces", timeout=_POST_TIMEOUT) if endpoint else None
        metric_exporter = (
            OTLPMetricExporter(endpoint=f"{endpoint}/v1/metrics", timeout=_POST_TIMEOUT) if endpoint else None
        )

        tracer_provider = TracerProvider(resource=resource, shutdown_on_exit=False)
        try:  # a processor already built must not keep its export thread when a later build fails
            if span_exporter is not None:
                queue_size = _positive_env("OTEL_BSP_MAX_QUEUE_SIZE", int, 2048, sys.maxsize)
                batch_size = _positive_env("OTEL_BSP_MAX_EXPORT_BATCH_SIZE", int, 512, sys.maxsize)
                if batch_size > queue_size:  # the SDK refuses a batch larger than the queue
                    logging.getLogger(__name__).warning(
                        "OTEL_BSP_MAX_EXPORT_BATCH_SIZE is above OTEL_BSP_MAX_QUEUE_SIZE; using the queue size %d",
                        queue_size,
                    )
                    batch_size = queue_size
                processor = BatchSpanProcessor(
                    span_exporter,
                    max_queue_size=queue_size,
                    schedule_delay_millis=_positive_env("OTEL_BSP_SCHEDULE_DELAY", int, 5000, _MAX_SDK_MILLIS),
                    max_export_batch_size=batch_size,
                )
                tracer_provider.add_span_processor(processor)
            readers = []
            if metric_exporter is not None:
                interval = _positive_env("OTEL_METRIC_EXPORT_INTERVAL", float, 60000.0, _MAX_SDK_MILLIS)
                readers.append(PeriodicExportingMetricReader(metric_exporter, export_interval_millis=interval))
            meter_provider = MeterProvider(resource=resource, metric_readers=readers, shutdown_on_exit=False)
        except BaseException:
            tracer_provider.shutdown()
            raise
        meter = meter_provider.get_meter("pipeline")
        instruments = Instruments(
            task_duration=meter.create_histogram("pipeline.task.duration", unit="s"),
            rows=meter.create_counter("pipeline.rows", unit="{row}"),
            check_failures=meter.create_counter("pipeline.check.failures", unit="{failure}"),
            connector_requests=meter.create_counter("pipeline.connector.requests", unit="{request}"),
            text_replaced=meter.create_counter("pipeline.connector.text_replaced", unit="{response}"),
            log_dropped=meter.create_counter("pipeline.log.dropped", unit="{record}"),
            slots_busy=meter.create_up_down_counter("pipeline.worker.slots_busy", unit="{slot}"),
            queue_wait=meter.create_histogram("pipeline.task.queue_wait", unit="s"),
            heartbeat_lag=meter.create_histogram("pipeline.lease.heartbeat_lag", unit="s"),
            lease_losses=meter.create_counter("pipeline.lease.losses", unit="{loss}"),
            lease_unwind=meter.create_histogram("pipeline.lease.unwind", unit="s"),
            unknown_jobs=meter.create_counter("pipeline.task.unknown_job", unit="{task}"),
            worker_paused=meter.create_counter("pipeline.worker.paused", unit="{pause}"),
        )

        # the logs-specific variable takes precedence, as for the SDK's own exporters
        headers = _valid_headers(
            parse_env_headers(
                os.environ.get("OTEL_EXPORTER_OTLP_LOGS_HEADERS") or os.environ.get("OTEL_EXPORTER_OTLP_HEADERS", ""),
                liberal=True,
            )
        )
        shipper = OtlpLogShipper(endpoint, resource, instruments.log_dropped, headers=headers) if endpoint else None
        queue_handler: logging.Handler | None = None
        if shipper is not None:
            queue_handler = _EagerQueueHandler(shipper.queue)
            queue_handler.addFilter(ContextFilter())
        stderr_handler = logging.StreamHandler()
        stderr_handler.setFormatter(JsonFormatter())
        stderr_handler.addFilter(ContextFilter())

        telemetry = Telemetry(
            tracer=tracer_provider.get_tracer("pipeline"),
            instruments=instruments,
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            log_shipper=shipper,
            queue_handler=queue_handler,
            stderr_handler=stderr_handler,
        )
        _active = telemetry  # before the first global side effect: OpenTelemetry ignores a second registration
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
        return telemetry
