import functools
import inspect
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any, cast


def reject_generators(fn: Callable[..., Any]) -> None:
    """Raise ``TypeError`` if ``fn`` is a generator function or an async generator function.

    Bound methods and ``functools.partial`` objects are unwrapped before the check. Any other object, including
    regular functions, coroutine functions and non-callables, is accepted without error.

    Preconditions:
        None.

    Args:
        fn: The object to check. Expected to be a callable.

    Returns:
        None.

    Raises:
        TypeError: If ``fn`` is, or after unwrapping a method or ``functools.partial`` resolves to, a generator
            function or an async generator function. The message names ``fn`` by its ``__qualname__``, or by its
            ``repr`` when it has none.

    """
    if inspect.isgeneratorfunction(fn) or inspect.isasyncgenfunction(fn):
        name = getattr(fn, "__qualname__", None) or repr(fn)
        raise TypeError(f"{name}: decorate the per-item function")


#: Callable type that takes a callable, a tuple of positional arguments and a dict of keyword arguments, and returns
#: a context manager. The value bound by the context manager's ``__enter__`` is typed as ``object``.
type CallContext = Callable[
    [
        Callable[..., Any],
        tuple[Any, ...],
        dict[str, Any],
    ],
    AbstractContextManager[object],
]


def around[F: Callable[..., Any]](make_context: CallContext) -> Callable[[F], F]:
    """Return a decorator that runs each call of the decorated function inside a context manager.

    The decorator builds a fresh context manager on every call by calling ``make_context(fn, args, kwargs)``, with the
    positional arguments as a tuple and the keyword arguments as a dict. Coroutine functions get an ``async`` wrapper
    that keeps the context open across the ``await``; every other callable gets a synchronous wrapper. If the context
    manager suppresses an exception raised by the call, the wrapper raises ``RuntimeError`` chained to the suppressed
    exception instead of returning ``None``.

    Preconditions:
        ``make_context`` returns a synchronous context manager on every call.

    Args:
        make_context: Factory called as ``make_context(fn, args, kwargs)`` on every call of the decorated function.

    Returns:
        A decorator that takes a callable ``fn`` and returns a wrapper of the same type, with ``fn``'s metadata
        copied by ``functools.wraps``.

    Raises:
        None.

    """

    def decorate(fn: F) -> F:
        """Wrap ``fn`` so that each call runs inside a context from ``make_context``.

        Preconditions:
            None.

        Args:
            fn: The callable to wrap.

        Returns:
            ``async_wrapper`` if ``inspect.iscoroutinefunction(fn)`` is true, otherwise ``wrapper``, cast to ``F``.

        Raises:
            TypeError: If ``fn`` is a generator function or an async generator function (raised by
                ``reject_generators``).

        """
        reject_generators(fn)

        def swallowed() -> RuntimeError:
            """Build the error raised when the context manager suppresses an exception from ``fn``.

            Preconditions:
                None.

            Args:
                None.

            Returns:
                A ``RuntimeError`` whose message names ``fn`` by its ``__qualname__``, or by ``repr(fn)`` if it has
                none.

            Raises:
                None.

            """
            name = getattr(fn, "__qualname__", repr(fn))
            return RuntimeError(f"{name}: the context manager swallowed an exception")

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                """Await ``fn`` inside a fresh context from ``make_context``.

                The final ``raise`` is reached only when the context manager suppressed an exception from ``fn``.

                Preconditions:
                    Called from a running event loop.

                Args:
                    *args: Positional arguments, passed to ``make_context`` as a tuple and to ``fn`` unchanged.
                    **kwargs: Keyword arguments, passed to ``make_context`` as a dict and then unpacked into ``fn``.

                Returns:
                    The value ``fn`` returns.

                Raises:
                    RuntimeError: The context manager suppressed an exception from ``fn``; that exception is the
                        ``__cause__``.
                    BaseException: Any exception from ``make_context``, from the context's ``__enter__`` or
                        ``__exit__``, or from ``fn`` and not suppressed, propagates unchanged.

                """
                caught: BaseException | None = None
                with make_context(fn, args, kwargs):
                    try:
                        return await fn(*args, **kwargs)
                    except BaseException as exc:
                        caught = exc
                        raise
                # noinspection PyUnreachableCode
                raise swallowed() from caught

            return cast(F, async_wrapper)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Call ``fn`` inside a fresh context from ``make_context``.

            The final ``raise`` is reached only when the context manager suppressed an exception from ``fn``.

            Preconditions:
                None.

            Args:
                *args: Positional arguments, passed to ``make_context`` as a tuple and to ``fn`` unchanged.
                **kwargs: Keyword arguments, passed to ``make_context`` as a dict and then unpacked into ``fn``.

            Returns:
                The value ``fn`` returns.

            Raises:
                RuntimeError: The context manager suppressed an exception from ``fn``; that exception is the
                    ``__cause__``.
                BaseException: Any exception from ``make_context``, from the context's ``__enter__`` or ``__exit__``,
                    or from ``fn`` and not suppressed, propagates unchanged.

            """
            caught: BaseException | None = None
            with make_context(fn, args, kwargs):
                try:
                    return fn(*args, **kwargs)
                except BaseException as exc:
                    caught = exc
                    raise
            # noinspection PyUnreachableCode
            raise swallowed() from caught

        return cast(F, wrapper)

    return decorate
