"""The pipeline's error types, how errors map to outcomes, and the decorators that translate and retry connector errors.

Public API:
    Outcome: The outcome reported for a task.
    PipelineError: Base class for pipeline errors, carrying an outcome and an error class name.
    TransientError: A failure that can succeed if retried later.
    RateLimited: A ``TransientError`` raised when a rate limit was exceeded, with an optional retry delay.
    AuthBlocked: A failure that needs a person to fix access before it can succeed.
    PermanentError: A failure that only a code or configuration change fixes.
    QualityFailed: A ``PermanentError`` raised when a batch had rows quarantined.
    LeaseConflict: Another worker holds the task; nothing is reported.
    AlreadyReported: The task's outcome was already recorded under this lease; nothing is reported.
    classify: Map an exception to an outcome and an error class name.
    ConnectorResponseError: An error response returned by an upstream connector.
    NothingLoaded: A ``PermanentError`` raised when an empty pruning load would remove or zero every row of the source.
    TableMismatch: A ``RuntimeError`` raised when a load target does not match its warehouse table.
    MAX_RETRY_AFTER: The longest retry delay the pipeline honours or stores: six hours.
    retry_after_from: Return the delay requested by a ``Retry-After`` header, at most ``MAX_RETRY_AFTER``.
    translates_errors: Return a decorator that converts connector and network errors into pipeline errors.
    retries: Build a decorator that retries an async function when it raises ``TransientError``.
    DatabaseUnavailable: A ``TransientError`` raised when the warehouse cannot be reached.
    DatabaseConflict: A ``TransientError`` raised when the warehouse aborted a transaction because of a conflict.
    SchemaMismatch: A ``RuntimeError`` raised when the warehouse schema revision is missing, malformed or too old.
    Interrupted: A ``TransientError`` reported when a task is stopped by worker shutdown.
    StartupRefused: A ``RuntimeError`` raised when a worker start-up check fails.
    UnknownJob: A ``PermanentError`` for a task whose job is not in the job registry.
    JobNotRunnable: A ``PipelineError`` with outcome ``UNEXPECTED`` for a known job that cannot be run.
    ClientErrorUnhandled: A ``PipelineError`` with outcome ``UNEXPECTED`` for a connector response no rule handles.
    RowLimitReached: A ``PermanentError`` raised when an endpoint reaches its row limit and cannot split the frame.
    TokenRejected: An ``AuthBlocked`` raised when a connector rejects the credential sent with a request.
    db_errors: Context manager that converts warehouse connection and conflict errors into pipeline errors.

"""

import asyncio
import functools
import inspect
import logging
import math
import random
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from enum import StrEnum
from typing import Any, ClassVar, Final

import httpx
import psycopg
import psycopg_pool
from opentelemetry import trace

from pipeline._wrapping import around, reject_generators


class Outcome(StrEnum):
    """Enumerate the possible outcomes of an operation.

    Each member is a string whose value is its lowercase name. Members compare equal to their values and hash like
    them.

    Attributes:
        SUCCEEDED: The task completed.
        RETRYABLE: A transient failure; trying again later can succeed.
        BLOCKED: A person must fix access (for example a source's authorisation) before it can succeed.
        FAILED: Only a code or configuration change fixes it.
        UNEXPECTED: An unclassified error; treat it as a bug.

    """

    SUCCEEDED = "succeeded"
    RETRYABLE = "retryable"
    BLOCKED = "blocked"
    FAILED = "failed"
    UNEXPECTED = "unexpected"


class PipelineError(Exception):
    """Represent a pipeline failure that carries a class-level outcome and error class name.

    ``outcome`` is ``Outcome.UNEXPECTED`` unless a subclass assigns another value, which may be ``None``.
    ``error_class`` is ``"PipelineError"`` on this class. A subclass that assigns ``error_class`` in its own body
    keeps that value; any other subclass gets its own ``__name__``, including subclasses of a class that assigned it.

    Attributes:
        outcome (Outcome | None): The outcome reported for this error. ``None`` means report nothing, and
            ``classify`` returns ``None``.
        error_class (str): The subclass's ``__name__`` unless the subclass assigns it.

    """

    outcome: ClassVar[Outcome | None] = Outcome.UNEXPECTED  # None: report nothing
    error_class: ClassVar[str] = "PipelineError"

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Set ``error_class`` on a new subclass to its ``__name__`` unless the subclass body assigns it.

        Preconditions:
            None.

        Args:
            **kwargs: Class keyword arguments, passed unchanged to ``super().__init_subclass__``.

        Returns:
            None.

        Raises:
            TypeError: If ``kwargs`` is not empty and no class later in the MRO accepts it, since
                ``object.__init_subclass__`` rejects all keyword arguments.

        """
        super().__init_subclass__(**kwargs)
        if "error_class" not in vars(cls):
            cls.error_class = cls.__name__


class TransientError(PipelineError):
    """Signal a pipeline failure that can be retried.

    Attributes:
        outcome (Outcome): ``Outcome.RETRYABLE``.

    """

    outcome: ClassVar[Outcome] = Outcome.RETRYABLE


class RateLimited(TransientError):
    """Signal that an operation was rejected because a rate limit was exceeded.

    Carries an optional hint for how long to wait before retrying.

    Attributes:
        retry_after (timedelta | None): Suggested delay before retrying, at most ``MAX_RETRY_AFTER``, or ``None`` if
            unknown.

    """

    def __init__(self, message: str, *, retry_after: timedelta | None = None) -> None:
        """Initialise the error with a message and an optional retry delay.

        Passes ``message`` to ``TransientError.__init__`` and stores ``retry_after`` on the instance as
        ``self.retry_after``, clamped to ``MAX_RETRY_AFTER`` as ``retry_after_from`` does.

        Preconditions:
            None.

        Args:
            message: Human-readable description of the rate-limit event.
            retry_after: Keyword-only. Suggested delay before retrying, at least zero, or ``None`` if unknown. A delay
                above ``MAX_RETRY_AFTER`` is stored as ``MAX_RETRY_AFTER``.

        Returns:
            None.

        Raises:
            ValueError: If ``retry_after`` is negative. The check runs before ``TransientError.__init__``.

        """
        if retry_after is not None and retry_after < timedelta(0):
            raise ValueError(f"retry_after must be at least zero, got {retry_after}")
        super().__init__(message)
        if retry_after is not None:
            retry_after = min(retry_after, MAX_RETRY_AFTER)
        self.retry_after = retry_after


class AuthBlocked(PipelineError):
    """Signal that a pipeline step was blocked by an authentication failure.

    The exception message is ``f"{connector}: {reason}"``. The class attribute ``outcome`` is ``Outcome.BLOCKED``, and
    the constructor arguments are stored unchanged as ``connector`` and ``reason``. Pickling and copying rebuild the
    instance from ``connector`` and ``reason`` by keyword and restore its ``__dict__``.

    Attributes:
        outcome (Outcome): ``Outcome.BLOCKED``.
        connector (str): The connector that reported the block, for example ``"tiktok_shop"``.
        reason (str): Explanation of why access was blocked.

    """

    outcome = Outcome.BLOCKED

    def __init__(self, *, connector: str, reason: str) -> None:
        """Initialise the error from the blocked connector and the reason.

        Passes the single message ``f"{connector}: {reason}"`` to ``PipelineError.__init__`` and stores both
        arguments unchanged.

        Preconditions:
            None. The signature makes both arguments keyword-only.

        Args:
            connector: The connector that reported the block, for example ``"tiktok_shop"``.
            reason: Explanation of why access was blocked.

        Returns:
            None.

        Raises:
            TypeError: If ``connector`` or ``reason`` is missing or passed positionally.

        """
        super().__init__(f"{connector}: {reason}")
        self.connector = connector
        self.reason = reason

    def __reduce__(self) -> tuple[object, ...]:
        """Return the pickling recipe that rebuilds this exception from its keyword arguments.

        Preconditions:
            ``self.connector`` and ``self.reason`` are set, and ``type(self)`` is importable under its own name.

        Args:
            None.

        Returns:
            The tuple
            ``(functools.partial(type(self), connector=self.connector, reason=self.reason), (), self.__dict__)``.
            Unpickling calls the partial with no arguments, then restores ``__dict__``, including any ``__notes__``.
            ``__cause__``, ``__context__`` and ``__traceback__`` are not preserved.

        Raises:
            None.

        """
        return (
            functools.partial(type(self), connector=self.connector, reason=self.reason),
            (),
            self.__dict__,
        )


class PermanentError(PipelineError):
    """Represent a pipeline error whose outcome is ``Outcome.FAILED``.

    Attributes:
        outcome (Outcome): ``Outcome.FAILED``.

    """

    outcome: ClassVar[Outcome] = Outcome.FAILED


class QualityFailed(PermanentError):
    """Signal that a batch failed its data-quality check because rows were quarantined.

    The message reads ``"<rows_quarantined> of <rows_total> rows quarantined"``, followed by ``": "`` and each
    ``<rule> <count>``, comma-separated, when ``by_rule`` is not empty, e.g. ``"4 of 50 rows quarantined: schema 3,
    total 1"``. Without ``by_rule`` it is unchanged, e.g. ``"3 of 10 rows quarantined"``. Both counts are stored as
    ``rows_total`` and ``rows_quarantined``, alongside ``by_rule``, so handlers can read them without parsing the
    message.

    Attributes:
        rows_total (int): Number of rows checked.
        rows_quarantined (int): Number of those rows that were quarantined.
        by_rule (dict[str, int]): Held records per rule, sorted by rule; empty when none is given. A record that
            broke two rules counts under both.

    """

    def __init__(self, *, rows_total: int, rows_quarantined: int, by_rule: Mapping[str, int] | None = None) -> None:
        """Build the exception message and record the row counts and per-rule counts.

        Preconditions:
            ``1 <= rows_quarantined <= rows_total`` and every ``by_rule`` count is from zero to ``rows_quarantined``;
            all are checked.

        Args:
            rows_total: Number of rows checked; at least one.
            rows_quarantined: Number of those rows that were quarantined; at least one and at most ``rows_total``.
            by_rule: Held records per rule, each count from zero to ``rows_quarantined``; ``None`` means none given.
                Copied into a new dict sorted by rule.

        Returns:
            None.

        Raises:
            ValueError: If ``rows_total`` is negative, ``rows_quarantined`` is below one or exceeds ``rows_total``, or
                a ``by_rule`` count is negative or exceeds ``rows_quarantined``.

        """
        if rows_total < 0:
            raise ValueError(f"rows_total must be at least zero, got {rows_total}")
        if rows_quarantined < 1:
            raise ValueError(f"rows_quarantined must be at least one, got {rows_quarantined}")
        if rows_quarantined > rows_total:
            raise ValueError(f"rows_quarantined ({rows_quarantined}) exceeds rows_total ({rows_total})")
        negative = sorted(rule for rule, count in (by_rule or {}).items() if count < 0)
        if negative:
            raise ValueError(f"by_rule counts must be at least zero, negative for: {', '.join(negative)}")
        excess = sorted(rule for rule, count in (by_rule or {}).items() if count > rows_quarantined)
        if excess:
            raise ValueError(f"by_rule counts exceed rows_quarantined ({rows_quarantined}) for: {', '.join(excess)}")
        counts = dict(sorted((by_rule or {}).items()))
        detail = ", ".join(f"{rule} {count}" for rule, count in counts.items())
        super().__init__(f"{rows_quarantined} of {rows_total} rows quarantined" + (f": {detail}" if detail else ""))
        self.rows_total = rows_total
        self.rows_quarantined = rows_quarantined
        self.by_rule = counts  # held records per rule; a record that broke two rules counts under both

    def __reduce__(self) -> tuple[object, ...]:
        """Return the recipe that ``pickle`` and ``copy`` use to rebuild this exception.

        Preconditions:
            The class is importable at module level under its own name.

        Args:
            None.

        Returns:
            A ``(callable, (), state)`` tuple:
            - ``callable`` is ``type(self)`` with ``rows_total``, ``rows_quarantined`` and ``by_rule`` bound as
              keyword arguments.
            - ``state`` is a shallow copy of ``self.__dict__``, which preserves extra attributes such as ``__notes__``.

            Copies and pickles therefore keep args, the counts, ``by_rule`` and any notes.

        Raises:
            None.

        """
        rebuild = functools.partial(
            type(self), rows_total=self.rows_total, rows_quarantined=self.rows_quarantined, by_rule=self.by_rule
        )
        return rebuild, (), self.__dict__.copy()


class LeaseConflict(PipelineError):
    """Represent a lease conflict as a ``PipelineError`` whose class-level ``outcome`` is ``None``.

    Construction, message handling and all other behaviour are inherited unchanged from ``PipelineError``.

    Attributes:
        outcome (None): ``None``, meaning another worker holds the task and nothing is reported (``classify``
            returns ``None``).

    """

    outcome = None


class AlreadyReported(PipelineError):
    """Represent an outcome already recorded under the caller's lease, as a ``PipelineError`` with ``outcome`` ``None``.

    Raised in place of ``LeaseConflict`` when the lease is no longer held because an outcome was recorded under the
    same lease token: the outcome has landed, so nothing is reported and no lease loss is recorded. Not a
    ``LeaseConflict``. Construction, message handling and all other behaviour are inherited unchanged from
    ``PipelineError``.

    Attributes:
        outcome (None): ``None``, meaning nothing is reported (``classify`` returns ``None``).

    """

    outcome = None


def classify(err: Exception) -> tuple[Outcome, str] | None:
    """Map an exception to an outcome and an error class name.

    A ``PipelineError`` is classified by its own ``outcome`` and ``error_class`` attributes. Any other value is
    classified as ``Outcome.UNEXPECTED`` with the name ``"Unexpected"``.

    Preconditions:
        ``err`` is an exception instance. This is not checked; any value that is not a ``PipelineError``, including
        ``None``, is classified as unexpected.

    Args:
        err: The exception to classify.

    Returns:
        ``(err.outcome, err.error_class)`` if ``err`` is a ``PipelineError`` whose ``outcome`` is not ``None``.
        ``None`` if ``err`` is a ``PipelineError`` whose ``outcome`` is ``None``.
        ``(Outcome.UNEXPECTED, "Unexpected")`` for any other ``err``.

    Raises:
        None.

    """
    if isinstance(err, PipelineError):
        if err.outcome is None:
            return None
        return err.outcome, err.error_class
    return Outcome.UNEXPECTED, "Unexpected"


class ConnectorResponseError(Exception):
    """An error response returned by an upstream connector.

    ``str(self)`` and the single item in ``self.args`` are the formatted message
    ``"{connector}: {status} code={code} path={path} req={request_id}: {message}"``. ``None`` values appear as
    ``None`` in that text. Instances survive ``copy.copy``, ``copy.deepcopy`` and ``pickle`` with ``args`` and all
    attributes intact when ``headers`` and any extra attribute can be copied and pickled, as ``httpx.Headers`` can; a
    ``types.MappingProxyType`` cannot be pickled. A subclass that adds constructor arguments must override
    ``__reduce__`` to pass them back, because the inherited one rebuilds the instance from this class's arguments only.

    Attributes:
        connector (str): Name of the connector that returned the error.
        path (str): Request path that produced the error.
        status (int): Status code of the response.
        code (int | None): Connector-specific error code, or ``None`` if the response had none.
        message (str): Error message reported by the connector, without the formatted prefix.
        request_id (str | None): Request identifier reported by the connector, or ``None`` if absent.
        headers (Mapping[str, str]): Response headers; the same object that was passed to ``__init__``.

    """

    def __init__(
        self,
        *,
        connector: str,
        path: str,
        status: int,
        code: int | None,
        message: str,
        request_id: str | None,
        headers: Mapping[str, str],
    ) -> None:
        """Initialise the error from the parts of a connector response.

        Preconditions:
            None.

        Args:
            connector: Name of the connector that returned the error.
            path: Request path that produced the error.
            status: Status code of the response.
            code: Connector-specific error code, or ``None`` if the response had none.
            message: Error message reported by the connector.
            request_id: Request identifier reported by the connector, or ``None`` if absent.
            headers: Response headers. Stored by reference, not copied.

        Returns:
            None.

        Raises:
            None.

        """
        super().__init__(f"{connector}: {status} code={code} path={path} req={request_id}: {message}")
        self.connector = connector
        self.path = path
        self.status = status
        self.code = code
        self.message = message
        self.request_id = request_id
        self.headers = headers

    def __reduce__(self) -> tuple[object, ...]:
        """Return the state needed to copy or pickle this exception.

        Preconditions:
            Every attribute in ``self.__dict__`` must be copyable, and picklable when pickling.

        Args:
            None.

        Returns:
            A ``(callable, (), state)`` tuple, where ``callable`` is ``type(self)`` with every constructor argument
            bound by keyword, and ``state`` is a shallow copy of ``self.__dict__``, which keeps extra attributes such
            as ``__notes__``.

        Raises:
            None.

        """
        rebuild = functools.partial(
            type(self),
            connector=self.connector,
            path=self.path,
            status=self.status,
            code=self.code,
            message=self.message,
            request_id=self.request_id,
            headers=self.headers,
        )
        return rebuild, (), self.__dict__.copy()


class NothingLoaded(PermanentError):
    """Signal that a load was refused because a pruning target had no rows.

    Adds no constructor or attributes of its own; the raiser supplies the message. ``stages.load.load`` raises it
    inside the load transaction, before any target is written, in two cases when pruning is on: a ``Snapshot``
    target with no rows, which would remove every row of the source (message ``"<table>: a snapshot with no rows
    would remove every row of the source"``), and an ``Update`` target with ``missing == "zero"`` and no rows,
    which would zero every row of the source (message ``"<table>: an update with no rows would zero every row of
    the source"``).

    Attributes:
        outcome (Outcome): ``Outcome.FAILED``, inherited from ``PermanentError``.

    """

    pass


class TableMismatch(RuntimeError):  # not a PipelineError, like SchemaMismatch: classify reports it as unexpected
    """Signal that a load target does not match its warehouse table, or the table is missing.

    Raised by ``stages.load._columns_checked`` inside the load transaction, before anything is written. The message
    lists every mismatch, separated by ``"; "``.

    Deliberately not a ``PipelineError``: retrying cannot fix it, so ``classify`` reports it as
    ``(Outcome.UNEXPECTED, "Unexpected")`` and it is treated as a bug.

    """

    pass


#: The longest retry delay the pipeline honours or stores. ``retry_after_from`` clamps to it, ``TaskReport`` refuses a
#: longer ``retry_after`` and ``control.report_outcome`` clamps the wait to it too.
MAX_RETRY_AFTER: Final = timedelta(hours=6)


def retry_after_from(headers: Mapping[str, str], *, now: datetime) -> timedelta | None:
    """Return the delay requested by a ``Retry-After`` header.

    The header name is matched case-insensitively, and the first matching entry in iteration order is used. A header
    sent more than once arrives as one entry joining the values with ``", "``, and only the first value is read: the
    text before the first comma when it is made only of ASCII digits, read as a number of seconds; otherwise the text
    before the second comma (an HTTP date holds one comma, after the weekday), parsed as an HTTP date and measured from
    ``now``. A date with no time zone is taken to be UTC. Whitespace around the value read is ignored. A delay longer
    than ``MAX_RETRY_AFTER`` is clamped to it.

    Preconditions:
        ``now`` is timezone-aware.

    Args:
        headers: Response headers, mapping each header name to its value.
        now: The moment an HTTP-date value is measured from.

    Returns:
        The requested delay as a ``timedelta`` from ``timedelta(0)`` to ``MAX_RETRY_AFTER``. An HTTP date at or before
        ``now`` gives ``timedelta(0)``; a longer delay, a number of seconds too large for ``timedelta`` included, gives
        ``MAX_RETRY_AFTER``. ``None`` only when there is no ``Retry-After`` header, or when its value is neither
        unsigned ASCII digits nor a parseable HTTP date.

    Raises:
        TypeError: If the value is an HTTP date and ``now`` is naive.

        ``ValueError`` and ``OverflowError`` from reading the number of seconds are caught, and the function returns
        ``MAX_RETRY_AFTER``. ``TypeError`` and ``ValueError`` from ``parsedate_to_datetime`` are caught, and the
        function returns ``None``.

    """
    value = next((v for k, v in headers.items() if k.lower() == "retry-after"), None)
    if value is None:
        return None
    seconds = value.split(",", 1)[0].strip()
    if seconds.isascii() and seconds.isdigit():
        try:
            return min(timedelta(seconds=int(seconds)), MAX_RETRY_AFTER)
        except ValueError, OverflowError:
            return MAX_RETRY_AFTER
    try:
        when = parsedate_to_datetime(",".join(value.split(",", 2)[:2]).strip())
    except TypeError, ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return min(max(when - now, timedelta(0)), MAX_RETRY_AFTER)


def translates_errors[F: Callable[..., Any]](
    connector: str,
    from_response: Callable[[ConnectorResponseError], PipelineError],
) -> Callable[[F], F]:
    """Return a decorator that converts errors from the decorated callable into pipeline errors.

    Each call to the decorated callable runs inside ``translate`` through ``around``. A ``ConnectorResponseError``
    is replaced by ``from_response(err)``. An ``httpx.TimeoutException``, ``httpx.NetworkError``,
    ``httpx.RemoteProtocolError`` or ``httpx.DecodingError`` is replaced by a ``TransientError`` with the message
    ``"{connector}: {exception type name}: {original message}"``. Every other exception, including any other
    ``PipelineError``, propagates unchanged. A replacement exception has the original as its ``__cause__``.

    Preconditions:
        ``from_response`` returns an exception instance and does not raise.

    Args:
        connector: Name of the upstream connector, used as the prefix of every ``TransientError`` message.
        from_response: Builds the ``PipelineError`` that replaces a ``ConnectorResponseError``.

    Returns:
        The decorator ``around(translate)``. It takes a callable of type ``F`` and returns a wrapped callable
        of the same type, whose calls raise the replacement errors described above. Applying the decorator to
        anything but a plain or coroutine function or method, or to a generator function or async generator
        function, raises ``TypeError`` (from ``reject_generators``); a plain one's call raises ``TypeError`` when it
        returns an awaitable, a generator or an async generator (see ``around``).

    Raises:
        None.

    """

    @contextmanager
    def translate(_fn: Callable[..., Any], _args: tuple[Any, ...], _kwargs: dict[str, Any]) -> Iterator[None]:
        """Convert an exception raised inside the managed block into a pipeline error.

        Preconditions:
            Used only as the context-manager factory passed to ``around``, which calls it with the wrapped
            callable and its arguments.

        Args:
            _fn: The wrapped callable. Unused.
            _args: Positional arguments of the call. Unused.
            _kwargs: Keyword arguments of the call. Unused.

        Yields:
            None, once.

        Raises:
            PipelineError: The result of ``from_response(err)`` when the block raises ``ConnectorResponseError``.
            TransientError: When the block raises ``httpx.TimeoutException``, ``httpx.NetworkError``,
                ``httpx.RemoteProtocolError`` or ``httpx.DecodingError``.
            Exception: Any other exception from the block propagates unchanged.

        """
        try:
            yield
        except ConnectorResponseError as err:
            raise from_response(err) from err
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError, httpx.DecodingError) as err:
            raise TransientError(f"{connector}: {type(err).__name__}: {err}") from err

    return around(translate)


def retries[**P, R](
    *,
    max_retries: int = 5,
    base: float = 1.0,
    cap: float = 60.0,
    jitter: float = 0.5,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Build a decorator that retries an async function when it raises ``TransientError``.

    The wrapped function is called once, then retried up to ``max_retries`` more times while it raises
    ``TransientError``. Before retry ``n`` (counting from 0), it waits ``min(base * 2**n, cap)`` seconds plus a
    uniform random value in ``[0, jitter]``, with the exponent clamped at 62, so a wait can exceed ``cap`` by up
    to ``jitter``. For a ``RateLimited`` error with a ``retry_after``, the wait is at least ``retry_after``. Each
    retry adds a ``retry`` event to the current trace span and logs a warning on this module's logger.

    Preconditions:
        None. Argument ranges are checked; see Raises.

    Args:
        max_retries: Number of retries after the first call, so the function runs at most ``max_retries + 1``
            times. ``0`` disables retrying.
        base: Wait in seconds before the first retry, before jitter.
        cap: Upper bound in seconds on the exponential part of each wait, and the largest ``retry_after`` that
            is waited out rather than re-raised.
        jitter: Upper bound in seconds on the random value added to each wait.
        sleep: Awaitable called with each wait in seconds.

    Returns:
        A decorator that takes an async function and returns a wrapper with the same signature.

    Raises:
        ValueError: If ``max_retries`` is negative, or ``base``, ``cap`` or ``jitter`` is negative, NaN or
            infinite.

    """
    if max_retries < 0:
        raise ValueError(f"max_retries must be >= 0, got {max_retries}")
    if not all(math.isfinite(value) and value >= 0 for value in (base, cap, jitter)):
        raise ValueError(f"base, cap and jitter must be finite and >= 0, got {base}, {cap}, {jitter}")

    log = logging.getLogger(__name__)

    def decorate(fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        """Wrap ``fn`` so that it is retried when it raises ``TransientError``.

        Preconditions:
            None.

        Args:
            fn: Async function to wrap.

        Returns:
            An async wrapper with ``fn``'s metadata copied by ``functools.wraps``.

        Raises:
            TypeError: If ``fn`` is not a coroutine function.
            Any exception ``reject_generators`` raises for ``fn``, unchanged.

        """
        reject_generators(fn)
        name = getattr(fn, "__qualname__", repr(fn))
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"retries needs an async function, got {name}")

        @functools.wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            """Await the wrapped function, retrying it with backoff on ``TransientError``.

            Preconditions:
                None.

            Args:
                *args: Positional arguments passed unchanged to every call.
                **kwargs: Keyword arguments passed unchanged to every call.

            Returns:
                The result of the first call that does not raise.

            Raises:
                TransientError: The error from the last call, once ``max_retries`` retries are used up.
                RateLimited: Immediately, without further retries, if its ``retry_after`` exceeds ``cap`` seconds.
                Any other exception from the wrapped function or from ``sleep``, including
                    ``asyncio.CancelledError``, propagates immediately without a retry.

            """
            attempt = 0
            while True:
                try:
                    return await fn(*args, **kwargs)
                except TransientError as err:
                    if attempt >= max_retries:
                        raise
                    # Exponential backoff, capped, then jitter.
                    wait = min(base * 2 ** min(attempt, 62), cap) + random.uniform(0, jitter)  # noqa: S311
                    if isinstance(err, RateLimited) and err.retry_after is not None:
                        retry_after = err.retry_after.total_seconds()
                        if retry_after > cap:
                            # Too long to hold a worker slot; Orchestrator requeues.
                            raise
                        wait = max(wait, retry_after)
                    trace.get_current_span().add_event(
                        "retry", {"attempt": attempt + 1, "pipeline.error_class": err.error_class, "wait_s": wait}
                    )
                    log.warning("%s failed (%s), retry %d in %.1fs", name, err.error_class, attempt + 1, wait)
                await sleep(wait)
                attempt += 1

        return wrapper

    return decorate


class DatabaseUnavailable(TransientError):
    """Signal that the warehouse could not be reached or dropped the connection.

    Attributes:
        outcome (Outcome): ``Outcome.RETRYABLE``, inherited from ``TransientError``.

    """

    pass


class DatabaseConflict(TransientError):
    """Signal that the warehouse aborted a transaction because of a serialization failure or deadlock.

    Attributes:
        outcome (Outcome): ``Outcome.RETRYABLE``, inherited from ``TransientError``.

    """

    pass


class SchemaMismatch(RuntimeError):
    """Signal that the warehouse's ``alembic_version`` is missing, malformed or older than ``REQUIRED_REVISION``.

    Raised by ``db.pool.check_schema`` at start-up, before any task is claimed, when ``alembic_version`` does not
    hold exactly one row, its ``version_num`` is longer than the allowed length or is not a string of ASCII decimal
    digits, or the revision is below ``REQUIRED_REVISION``. A failure to query the table at all (for example a
    missing table) is not this error.

    Deliberately not a ``PipelineError``: retrying cannot fix it, so ``classify`` reports it as
    ``(Outcome.UNEXPECTED, "Unexpected")`` and it is treated as a bug.

    """

    pass


class Interrupted(TransientError):
    """Signal that a task was stopped before it finished because the worker is shutting down.

    The task is reported as retryable so another run can pick it up.

    Attributes:
        outcome (Outcome): ``Outcome.RETRYABLE``, inherited from ``TransientError``.

    """

    pass


class StartupRefused(RuntimeError):
    """Signal that the worker refused to start because a start-up check failed.

    Deliberately not a ``PipelineError``: no task is running yet, so there is no outcome to report. If it does reach
    ``classify``, it is reported as ``(Outcome.UNEXPECTED, "Unexpected")``.

    """

    pass


class UnknownJob(PermanentError):
    """Signal that a task names a job that is not in the job registry.

    Retrying cannot fix it; only a code or configuration change can.

    Attributes:
        outcome (Outcome): ``Outcome.FAILED``, inherited from ``PermanentError``.

    """

    pass


class JobNotRunnable(PipelineError):
    """Signal that a task names a known job that cannot be run.

    No subclass outcome is assigned, so ``classify`` reports it as ``(Outcome.UNEXPECTED, "JobNotRunnable")``.

    Attributes:
        outcome (Outcome): ``Outcome.UNEXPECTED``, inherited from ``PipelineError``.

    """

    pass


class ClientErrorUnhandled(PipelineError):
    """Signal a connector response the client cannot handle.

    Raised for an error code no rule matches, a 2xx response whose body is not JSON, or a body a spec cannot read.
    No outcome is assigned, so ``classify`` reports it as ``(Outcome.UNEXPECTED, "ClientErrorUnhandled")``. It is not
    a ``TransientError``, so ``retries`` does not retry it. Pickling and copying rebuild the instance from its
    keyword arguments and restore its ``__dict__``.

    Attributes:
        outcome (Outcome): ``Outcome.UNEXPECTED``, inherited from ``PipelineError``.
        connector (str): Name of the connector that returned the response.
        path (str): Request path that produced the response.
        status (int): Status code of the response.
        code (int | None): Connector-specific error code, or ``None`` if the response had none.
        message (str): Description of the error, without the formatted prefix.
        request_id (str | None): Request identifier reported by the connector, or ``None`` if absent.

    """

    def __init__(
        self,
        *,
        connector: str,
        path: str,
        status: int,
        code: int | None,
        message: str,
        request_id: str | None,
    ) -> None:
        """Build the message ``{connector}: unhandled {status} code={code} path={path} req={request_id}: {message}``.

        Stores every argument unchanged as an attribute of the same name. ``None`` values appear as ``None`` in the
        message.

        Preconditions:
            Every argument is passed by keyword.

        Args:
            connector: Name of the connector that returned the response.
            path: Request path that produced the response.
            status: Status code of the response.
            code: Connector-specific error code, or ``None`` if the response had none.
            message: Description of the error.
            request_id: Request identifier reported by the connector, or ``None`` if absent.

        Returns:
            None.

        Raises:
            TypeError: If an argument is missing or passed positionally.

        """
        super().__init__(f"{connector}: unhandled {status} code={code} path={path} req={request_id}: {message}")
        self.connector = connector
        self.path = path
        self.status = status
        self.code = code
        self.message = message
        self.request_id = request_id

    def __reduce__(self) -> tuple[object, ...]:
        """Return the recipe that ``pickle`` and ``copy`` use to rebuild this exception.

        Preconditions:
            The class is importable at module level under its own name.

        Args:
            None.

        Returns:
            A ``(callable, (), state)`` tuple:
            - ``callable`` is ``type(self)`` with ``connector``, ``path``, ``status``, ``code``, ``message`` and
              ``request_id`` bound as keyword arguments.
            - ``state`` is a shallow copy of ``self.__dict__``, which preserves extra attributes such as ``__notes__``.

        Raises:
            None.

        """
        rebuild = functools.partial(
            type(self),
            connector=self.connector,
            path=self.path,
            status=self.status,
            code=self.code,
            message=self.message,
            request_id=self.request_id,
        )
        return rebuild, (), self.__dict__.copy()


class RowLimitReached(PermanentError):
    """Signal that an endpoint reached its row limit and the fetch cannot split the time frame to fit.

    The message reads ``"<endpoint>: reached the <limit>-row limit and the fetch cannot split the time frame to
    fit"``. The wording covers both cases it is raised in: a reported total over the limit in a frame that cannot be
    halved further, and a connector without a total whose rows reach the limit, which the fetch never splits. Both
    arguments are stored as ``endpoint`` and ``limit`` so handlers can read them without parsing the message. Pickling
    and copying rebuild the instance from ``endpoint`` and ``limit`` by keyword and restore its ``__dict__``.

    Attributes:
        outcome (Outcome): ``Outcome.FAILED``, inherited from ``PermanentError``.
        endpoint (str): Name of the endpoint spec whose row limit was reached.
        limit (int): The endpoint's maximum number of rows per time frame.

    """

    def __init__(self, *, endpoint: str, limit: int) -> None:
        """Build the exception message and record the endpoint and its row limit.

        Preconditions:
            ``endpoint`` and ``limit`` are passed by keyword.

        Args:
            endpoint: Name of the endpoint spec whose row limit was reached.
            limit: The endpoint's maximum number of rows per time frame.

        Returns:
            None.

        Raises:
            TypeError: If ``endpoint`` or ``limit`` is missing or passed positionally.

        """
        super().__init__(f"{endpoint}: reached the {limit}-row limit and the fetch cannot split the time frame to fit")
        self.endpoint = endpoint
        self.limit = limit

    def __reduce__(self) -> tuple[object, ...]:
        """Return the recipe that ``pickle`` and ``copy`` use to rebuild this exception.

        Preconditions:
            The class is importable at module level under its own name.

        Args:
            None.

        Returns:
            A ``(callable, (), state)`` tuple:
            - ``callable`` is ``type(self)`` with ``endpoint`` and ``limit`` bound as keyword arguments.
            - ``state`` is a shallow copy of ``self.__dict__``, which preserves extra attributes such as ``__notes__``.

        Raises:
            None.

        """
        return functools.partial(type(self), endpoint=self.endpoint, limit=self.limit), (), self.__dict__.copy()


class TokenRejected(AuthBlocked):
    """Signal that a connector rejected the credential sent with a request.

    Raised when a connector error matches a ``TOKEN_REJECTED`` rule. ``HttpConnectorClient`` reports the credential back
    to its ``CredentialProvider`` before re-raising. It is not a ``TransientError``, so ``retries`` does not retry it.
    Construction, message, pickling and the ``Outcome.BLOCKED`` outcome are inherited unchanged from ``AuthBlocked``;
    ``classify`` reports it as ``(Outcome.BLOCKED, "TokenRejected")``.

    Attributes:
        outcome (Outcome): ``Outcome.BLOCKED``, inherited from ``AuthBlocked``.
        connector (str): Name of the connector that rejected the credential.
        reason (str): The connector's error message.

    """

    pass  # the connector rejected the credential itself; the client reports it to its CredentialProvider


#: Errors that mean the warehouse is unreachable or the connection was lost, including the server ending an idle
#: session (57P05) or an idle transaction (25P03).
_DB_UNAVAILABLE: tuple[type[Exception], ...] = (
    psycopg.errors.ConnectionException,
    psycopg.errors.ConnectionFailure,
    psycopg.errors.SqlclientUnableToEstablishSqlconnection,
    psycopg.errors.ConnectionDoesNotExist,
    psycopg.errors.AdminShutdown,
    psycopg.errors.CrashShutdown,
    psycopg.errors.CannotConnectNow,
    psycopg.errors.IdleSessionTimeout,
    psycopg.errors.IdleInTransactionSessionTimeout,
    psycopg_pool.PoolTimeout,
)
#: Errors that mean the transaction lost a conflict with another transaction.
_DB_CONFLICT: tuple[type[Exception], ...] = (psycopg.errors.SerializationFailure, psycopg.errors.DeadlockDetected)


@contextmanager
def db_errors() -> Iterator[None]:
    """Convert warehouse connection and conflict errors raised inside the block into pipeline errors.

    An error in ``_DB_CONFLICT`` becomes ``DatabaseConflict``. An error in ``_DB_UNAVAILABLE``, or an exception whose
    type is exactly ``psycopg.OperationalError``, becomes ``DatabaseUnavailable``. Each replacement has the message
    ``"warehouse: {exception type name}: {original message}"`` and the original as its ``__cause__``. Every other
    exception, including other ``psycopg.OperationalError`` subclasses, propagates unchanged.

    Preconditions:
        None.

    Args:
        None.

    Yields:
        None, once.

    Raises:
        DatabaseConflict: When the block raises an error in ``_DB_CONFLICT``.
        DatabaseUnavailable: When the block raises an error in ``_DB_UNAVAILABLE`` or a bare
            ``psycopg.OperationalError``.

    """
    try:
        yield
    except _DB_CONFLICT as err:
        raise DatabaseConflict(f"warehouse: {type(err).__name__}: {err}") from err
    except _DB_UNAVAILABLE as err:
        raise DatabaseUnavailable(f"warehouse: {type(err).__name__}: {err}") from err
    except psycopg.OperationalError as err:
        if type(err) is not psycopg.OperationalError:  # only the bare class: a dropped connection
            raise
        raise DatabaseUnavailable(f"warehouse: {type(err).__name__}: {err}") from err
