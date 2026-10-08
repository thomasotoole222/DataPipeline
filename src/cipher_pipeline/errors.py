"""The pipeline's error types, how errors map to outcomes, and the decorators that translate and retry source errors.

Public API:
    Outcome: The outcome reported for a task.
    PipelineError: Base class for pipeline errors, carrying an outcome and an error class name.
    TransientError: A failure that can succeed if retried later.
    RateLimited: A ``TransientError`` raised when a rate limit was exceeded, with an optional retry delay.
    AuthBlocked: A failure that needs a person to fix access before it can succeed.
    PermanentError: A failure that only a code or configuration change fixes.
    QualityFailed: A ``PermanentError`` raised when a batch had rows quarantined.
    LeaseConflict: Another worker holds the task; nothing is reported.
    classify: Map an exception to an outcome and an error class name.
    SourceResponseError: An error response returned by an upstream source.
    retry_after_from: Return the delay requested by a ``Retry-After`` header.
    translates_errors: Return a decorator that converts source and network errors into pipeline errors.
    retries: Build a decorator that retries an async function when it raises ``TransientError``.

"""

import asyncio
import functools
import inspect
import logging
import random
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from enum import StrEnum
from typing import Any, ClassVar

import httpx
from opentelemetry import trace

from cipher_pipeline._wrapping import around, reject_generators


class Outcome(StrEnum):
    """Enumerate the possible outcomes of an operation.

    Each member is a string whose value is its lowercase name. Members compare equal to their values and hash like
    them.

    Attributes:
        SUCCEEDED: The task completed.
        RETRYABLE: A transient failure; trying again later can succeed.
        BLOCKED: A person must fix access (for example a shop's authorisation) before it can succeed.
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

    outcome: ClassVar[Outcome | None] = Outcome.UNEXPECTED
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
        retry_after (timedelta | None): Suggested delay before retrying, or ``None`` if unknown.

    """

    def __init__(self, message: str, *, retry_after: timedelta | None = None) -> None:
        """Initialise the error with a message and an optional retry delay.

        Passes ``message`` to ``TransientError.__init__`` and stores ``retry_after`` on the instance as
        ``self.retry_after``.

        Preconditions:
            ``retry_after``, if given, is a ``datetime.timedelta`` greater than or equal to zero.

        Args:
            message: Human-readable description of the rate-limit event.
            retry_after: Keyword-only. Suggested delay before retrying, or ``None`` if unknown.

        Returns:
            None.

        Raises:
            None directly. Any exception raised by ``TransientError.__init__`` propagates.

        """
        super().__init__(message)
        self.retry_after = retry_after


class AuthBlocked(PipelineError):
    """Signal that a pipeline step was blocked by an authentication failure.

    The exception message is ``f"{source}: {reason}"``. The class attribute ``outcome`` is ``Outcome.BLOCKED``, and the
    constructor arguments are stored unchanged as ``source`` and ``reason``. Pickling and copying rebuild the instance
    from ``source`` and ``reason`` by keyword and restore its ``__dict__``.

    Attributes:
        outcome (Outcome): ``Outcome.BLOCKED``.
        source (str): Name of the component that reported the block.
        reason (str): Explanation of why access was blocked.

    """

    outcome = Outcome.BLOCKED

    def __init__(self, *, source: str, reason: str) -> None:
        """Initialise the error from the blocked source and the reason.

        Preconditions:
            ``source`` and ``reason`` are passed by keyword. ``PipelineError.__init__`` accepts a single positional
            message.

        Args:
            source: Name of the component that reported the block.
            reason: Explanation of why access was blocked.

        Returns:
            None.

        Raises:
            TypeError: If ``source`` or ``reason`` is missing or passed positionally.
            Exceptions raised by ``PipelineError.__init__`` propagate unchanged.

        """
        super().__init__(f"{source}: {reason}")
        self.source = source
        self.reason = reason

    def __reduce__(self) -> tuple[object, ...]:
        """Return the pickling recipe that rebuilds this exception from its keyword arguments.

        Preconditions:
            ``self.source`` and ``self.reason`` are set, and ``AuthBlocked`` is importable from its defining module.

        Args:
            None.

        Returns:
            The tuple ``(functools.partial(type(self), source=self.source, reason=self.reason), (), self.__dict__)``.
            Unpickling calls the partial with no arguments, then restores ``__dict__``, including any ``__notes__``.
            ``__cause__``, ``__context__`` and ``__traceback__`` are not preserved.

        Raises:
            None.

        """
        return (
            functools.partial(type(self), source=self.source, reason=self.reason),
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

    The message reads ``"<rows_quarantined> of <rows_total> rows quarantined"``. Both counts are stored as
    ``rows_total`` and ``rows_quarantined`` so handlers can read them without parsing the message.

    Attributes:
        rows_total (int): Number of rows checked.
        rows_quarantined (int): Number of those rows that were quarantined.

    """

    def __init__(self, *, rows_total: int, rows_quarantined: int) -> None:
        """Build the exception message and record the row counts.

        Preconditions:
            ``0 <= rows_quarantined <= rows_total``.

        Args:
            rows_total: Number of rows checked.
            rows_quarantined: Number of those rows that were quarantined.

        Returns:
            None.

        Raises:
            None.

        """
        super().__init__(f"{rows_quarantined} of {rows_total} rows quarantined")
        self.rows_total = rows_total
        self.rows_quarantined = rows_quarantined

    def __reduce__(self) -> tuple[object, ...]:
        """Return the recipe that ``pickle`` and ``copy`` use to rebuild this exception.

        Preconditions:
            The class is importable at module level under its own name.

        Args:
            None.

        Returns:
            A ``(callable, (), state)`` tuple:
            - ``callable`` is ``type(self)`` with ``rows_total`` and ``rows_quarantined`` bound as keyword arguments.
            - ``state`` is a shallow copy of ``self.__dict__``, which preserves extra attributes such as ``__notes__``.

        Raises:
            None.

        """
        rebuild = functools.partial(type(self), rows_total=self.rows_total, rows_quarantined=self.rows_quarantined)
        return rebuild, (), self.__dict__.copy()


class LeaseConflict(PipelineError):
    """Represent a lease conflict as a ``PipelineError`` whose class-level ``outcome`` is ``None``.

    Construction, message handling and all other behaviour are inherited unchanged from ``PipelineError``.

    Attributes:
        outcome (None): ``None``, meaning another worker holds the task and nothing is reported (``classify``
            returns ``None``).

    """

    outcome = None


def classify(exc: Exception) -> tuple[Outcome, str] | None:
    """Map an exception to an outcome and an error class name.

    A ``PipelineError`` is classified by its own ``outcome`` and ``error_class`` attributes. Any other value is
    classified as ``Outcome.UNEXPECTED`` with the name ``"Unexpected"``.

    Preconditions:
        ``exc`` is an exception instance. This is not checked; any value that is not a ``PipelineError``, including
        ``None``, is classified as unexpected.

    Args:
        exc: The exception to classify.

    Returns:
        ``(exc.outcome, exc.error_class)`` if ``exc`` is a ``PipelineError`` whose ``outcome`` is not ``None``.
        ``None`` if ``exc`` is a ``PipelineError`` whose ``outcome`` is ``None``.
        ``(Outcome.UNEXPECTED, "Unexpected")`` for any other ``exc``.

    Raises:
        None.

    """
    if isinstance(exc, PipelineError):
        if exc.outcome is None:
            return None
        return exc.outcome, exc.error_class
    return Outcome.UNEXPECTED, "Unexpected"


class SourceResponseError(Exception):
    """An error response returned by an upstream source.

    ``str(self)`` and the single item in ``self.args`` are the formatted message
    ``"{source}: {status} code={code} path={path} req={request_id}: {message}"``. ``None`` values appear as
    ``None`` in that text. Instances survive ``copy.copy``, ``copy.deepcopy`` and ``pickle`` with ``args`` and all
    attributes intact. Subclasses that add constructor arguments must store them as instance attributes so that
    ``__reduce__`` carries them.

    Attributes:
        source (str): Name of the source that returned the error.
        path (str): Request path that produced the error.
        status (int): Status code of the response.
        code (int | None): Source-specific error code, or ``None`` if the response had none.
        message (str): Error message reported by the source, without the formatted prefix.
        request_id (str | None): Request identifier reported by the source, or ``None`` if absent.
        headers (Mapping[str, str]): Response headers; the same object that was passed to ``__init__``.

    """

    def __init__(
        self,
        *,
        source: str,
        path: str,
        status: int,
        code: int | None,
        message: str,
        request_id: str | None,
        headers: Mapping[str, str],
    ) -> None:
        """Initialise the error from the parts of a source response.

        Preconditions:
            None.

        Args:
            source: Name of the source that returned the error.
            path: Request path that produced the error.
            status: Status code of the response.
            code: Source-specific error code, or ``None`` if the response had none.
            message: Error message reported by the source.
            request_id: Request identifier reported by the source, or ``None`` if absent.
            headers: Response headers. Stored by reference, not copied.

        Returns:
            None.

        Raises:
            None.

        """
        super().__init__(f"{source}: {status} code={code} path={path} req={request_id}: {message}")
        self.source = source
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
            source=self.source,
            path=self.path,
            status=self.status,
            code=self.code,
            message=self.message,
            request_id=self.request_id,
            headers=self.headers,
        )
        return rebuild, (), self.__dict__.copy()


def retry_after_from(headers: Mapping[str, str], *, now: datetime) -> timedelta | None:
    """Return the delay requested by a ``Retry-After`` header.

    The header name is matched case-insensitively, and the first matching entry in iteration order is used. A value
    made only of ASCII digits is read as a number of seconds. Any other value is parsed as an HTTP date and measured
    from ``now``; a date with no time zone is taken to be UTC. Surrounding whitespace in the value is ignored.

    Preconditions:
        ``now`` is timezone-aware.

    Args:
        headers: Response headers, mapping each header name to its value.
        now: The moment an HTTP-date value is measured from.

    Returns:
        The requested delay as a non-negative ``timedelta``. An HTTP date at or before ``now`` gives ``timedelta(0)``.
        ``None`` only when there is no ``Retry-After`` header, or when its value is neither unsigned ASCII digits nor a
        parseable HTTP date. A number of seconds too large for ``timedelta`` gives ``timedelta.max``.

    Raises:
        TypeError: If the value is an HTTP date and ``now`` is naive.

        ``ValueError`` and ``OverflowError`` from reading the number of seconds are caught, and the function returns
        ``timedelta.max``. ``TypeError`` and ``ValueError`` from ``parsedate_to_datetime`` are caught, and the
        function returns ``None``.

    """
    value = next((v for k, v in headers.items() if k.lower() == "retry-after"), None)
    if value is None:
        return None
    value = value.strip()
    if value.isascii() and value.isdigit():
        try:
            return timedelta(seconds=int(value))
        except ValueError, OverflowError:
            return timedelta.max
    try:
        when = parsedate_to_datetime(value)
    except TypeError, ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(when - now, timedelta(0))


def translates_errors[F: Callable[..., Any]](
    source: str,
    from_response: Callable[[SourceResponseError], PipelineError],
) -> Callable[[F], F]:
    """Return a decorator that converts errors from the decorated callable into pipeline errors.

    Each call to the decorated callable runs inside ``translate`` through ``around``. A ``SourceResponseError``
    is replaced by ``from_response(exc)``. An ``httpx.TimeoutException``, ``httpx.NetworkError`` or
    ``httpx.RemoteProtocolError`` is replaced by a ``TransientError`` with the message
    ``"{source}: {exception type name}: {original message}"``. Every other exception, including any other
    ``PipelineError``, propagates unchanged. A replacement exception has the original as its ``__cause__``.

    Preconditions:
        ``from_response`` returns an exception instance and does not raise.

    Args:
        source: Name of the upstream source, used as the prefix of every ``TransientError`` message.
        from_response: Builds the ``PipelineError`` that replaces a ``SourceResponseError``.

    Returns:
        The decorator ``around(translate)``. It takes a callable of type ``F`` and returns a wrapped callable
        of the same type, whose calls raise the replacement errors described above.

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
            PipelineError: The result of ``from_response(exc)`` when the block raises ``SourceResponseError``.
            TransientError: When the block raises ``httpx.TimeoutException``, ``httpx.NetworkError`` or
                ``httpx.RemoteProtocolError``.

        """
        try:
            yield
        except SourceResponseError as exc:
            raise from_response(exc) from exc
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
            raise TransientError(f"{source}: {type(exc).__name__}: {exc}") from exc

    return around(translate)


def retries[**P, R](
    *,
    attempts: int = 5,
    base: float = 1.0,
    cap: float = 60.0,
    jitter: float = 0.5,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Build a decorator that retries an async function when it raises ``TransientError``.

    The wrapped function is called once, then retried up to ``attempts`` more times while it raises
    ``TransientError``. Before retry ``n`` (counting from 0), it waits ``min(base * 2**n, cap)`` seconds plus a
    uniform random value in ``[0, jitter]``, with the exponent clamped at 62, so a wait can exceed ``cap`` by up
    to ``jitter``. For a ``RateLimited`` error with a ``retry_after``, the wait is at least ``retry_after``. Each
    retry adds a ``retry`` event to the current trace span and logs a warning on this module's logger.

    Preconditions:
        None. Argument ranges are checked; see Raises.

    Args:
        attempts: Number of retries after the first call. ``0`` disables retrying.
        base: Wait in seconds before the first retry, before jitter.
        cap: Upper bound in seconds on the exponential part of each wait, and the largest ``retry_after`` that
            is waited out rather than re-raised.
        jitter: Upper bound in seconds on the random value added to each wait.
        sleep: Awaitable called with each wait in seconds.

    Returns:
        A decorator that takes an async function and returns a wrapper with the same signature.

    Raises:
        ValueError: If ``attempts``, ``base``, ``cap`` or ``jitter`` is negative.

    """
    if attempts < 0:
        raise ValueError(f"attempts must be >= 0, got {attempts}")
    if base < 0 or cap < 0 or jitter < 0:
        raise ValueError(f"base, cap and jitter must be >= 0, got {base}, {cap}, {jitter}")

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
                TransientError: The error from the last call, once ``attempts`` retries are used up.
                RateLimited: Immediately, without further retries, if its ``retry_after`` exceeds ``cap`` seconds.
                Any other exception from the wrapped function or from ``sleep``, including
                    ``asyncio.CancelledError``, propagates immediately without a retry.

            """
            attempt = 0
            while True:
                try:
                    return await fn(*args, **kwargs)
                except TransientError as exc:
                    if attempt >= attempts:
                        raise
                    # Exponential backoff, capped, then jitter.
                    wait = min(base * 2 ** min(attempt, 62), cap) + random.uniform(0, jitter)  # noqa: S311
                    if isinstance(exc, RateLimited) and exc.retry_after is not None:
                        retry_after = exc.retry_after.total_seconds()
                        if retry_after > cap:
                            # Too long to hold a worker slot; Orchestrator requeues.
                            raise
                        wait = max(wait, retry_after)
                    trace.get_current_span().add_event(
                        "retry", {"attempt": attempt + 1, "hub.error_class": exc.error_class, "wait_s": wait}
                    )
                    log.warning("%s failed (%s), retry %d in %.1fs", name, exc.error_class, attempt + 1, wait)
                await sleep(wait)
                attempt += 1

        return wrapper

    return decorate
