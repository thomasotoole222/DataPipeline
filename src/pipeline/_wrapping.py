"""Decorator plumbing that runs each call of a function inside a per-call context manager.

Public API:
    reject_generators: Raise ``TypeError`` for a callable that ``around`` cannot wrap safely.
    CallContext: Type of the factory ``around`` calls to build a context manager for each call.
    around: Return a decorator that runs each call of the decorated function inside a context manager.

"""

import asyncio
import functools
import inspect
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any, cast


def reject_generators(fn: Callable[..., Any]) -> None:
    """Raise ``TypeError`` unless ``fn`` is a plain or coroutine function or method that is not a generator.

    ``functools.partial`` objects (nested ones included) and then a bound method are unwrapped; what remains must be
    a Python function (``inspect.isfunction``): a plain function or a coroutine function, but not a generator
    function or an async generator function. Every other callable is refused, such as an object whose
    ``__call__`` is ``async``, which ``inspect.iscoroutinefunction`` does not detect, so ``around`` would give it a
    synchronous wrapper that closes its context before the work is awaited, or a class or builtin.

    Preconditions:
        None.

    Args:
        fn: The object to check.

    Returns:
        None.

    Raises:
        TypeError: If ``fn``, after unwrapping partials and a bound method, is not a Python function, or is a
            generator function or an async generator function. The message names ``fn`` by its ``__qualname__``,
            or by its ``repr`` when it has none.

    """
    name = getattr(fn, "__qualname__", None) or repr(fn)
    target: object = fn
    while isinstance(target, functools.partial):
        target = target.func
    if inspect.ismethod(target):
        target = target.__func__
    if not inspect.isfunction(target):
        raise TypeError(f"{name}: decorate a function or method, not a {type(target).__name__}")
    if inspect.isgeneratorfunction(fn) or inspect.isasyncgenfunction(fn):
        raise TypeError(f"{name}: decorate the per-item function")


#: Callable type that takes a callable, a tuple of positional arguments and a dict of keyword arguments, and returns
#: a context manager. The value bound by the context manager's ``__enter__`` is typed as ``object``.
type CallContext = Callable[
    [Callable[..., Any], tuple[Any, ...], dict[str, Any]],
    AbstractContextManager[object],
]


def around[F: Callable[..., Any]](make_context: CallContext) -> Callable[[F], F]:
    """Return a decorator that runs each call of the decorated function inside a context manager.

    The decorator builds a fresh context manager on every call by calling ``make_context(fn, args, kwargs)``, with the
    positional arguments as a tuple and the keyword arguments as a dict. Only what ``reject_generators`` accepts can
    be decorated. Coroutine functions get an ``async`` wrapper that keeps the context open across the ``await``; plain
    functions and methods get a synchronous wrapper, which raises ``TypeError`` inside the context when a call
    returns deferred work: an awaitable (a coroutine, an ``asyncio.Future`` or ``asyncio.Task`` included), a
    generator or an async generator, since that work would run after the context closed. Every other result, an
    eager iterator such as ``io.StringIO`` or ``csv.reader`` included, is returned. If the context manager suppresses
    an exception raised by the call, the wrapper raises ``RuntimeError`` chained to the suppressed exception instead
    of returning ``None``.

    Preconditions:
        ``make_context`` returns a synchronous context manager on every call.

    Args:
        make_context: Factory called as ``make_context(fn, args, kwargs)`` on every call of the decorated function.

    Returns:
        A decorator that takes a callable ``fn`` and returns a wrapper of the same type, with ``fn``'s metadata
        copied by ``functools.wraps``. The decorator raises ``TypeError`` (from ``reject_generators``) for a
        callable it refuses.

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
            TypeError: If ``fn`` is not a plain or coroutine function or method, or is a generator function or an
                async generator function (raised by ``reject_generators``).

        """
        reject_generators(fn)
        name = getattr(fn, "__qualname__", None) or repr(fn)

        def swallowed() -> RuntimeError:
            """Build the error raised when the context manager suppresses an exception from ``fn``.

            Preconditions:
                None.

            Args:
                None.

            Returns:
                A ``RuntimeError`` whose message names ``fn`` by its ``__qualname__``, or by ``repr(fn)`` when it has
                no ``__qualname__`` or an empty one.

            Raises:
                None.

            """
            return RuntimeError(f"{name}: the context manager swallowed an exception")

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                """Await ``fn`` inside a fresh context from ``make_context``, which stays open across the ``await``.

                The result is not inspected. The final ``raise`` is reached only when the context manager suppressed
                an exception from ``fn``, ``asyncio.CancelledError`` included.

                Preconditions:
                    ``fn(*args, **kwargs)`` returns an awaitable; ``inspect.iscoroutinefunction(fn)`` being true
                    (checked by ``decorate``) is the only evidence of it.

                Args:
                    *args: Positional arguments, passed to ``make_context`` as a tuple and to ``fn`` unchanged.
                    **kwargs: Keyword arguments, passed to ``make_context`` as a dict and then unpacked into ``fn``.

                Returns:
                    The value the awaited result of ``fn`` produces.

                Raises:
                    RuntimeError: The context manager suppressed an exception from ``fn``; that exception is the
                        ``__cause__``.
                    BaseException: Any exception from ``make_context``, from the context's ``__enter__`` or
                        ``__exit__``, or from calling or awaiting ``fn`` and not suppressed, propagates unchanged.

                """
                caught: BaseException | None = None
                with make_context(fn, args, kwargs):
                    try:
                        return await fn(*args, **kwargs)
                    except BaseException as err:
                        caught = err
                        raise
                raise swallowed() from caught

            return cast(F, async_wrapper)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Call ``fn`` inside a fresh context from ``make_context``.

            A result that stands for deferred work (an awaitable, a generator or an async generator) is refused with
            ``TypeError`` raised inside the context. A refused coroutine is closed and a refused ``asyncio.Future``
            (an ``asyncio.Task`` included) is cancelled first; any other refused result is left as it is. Every other
            result, an eager iterator such as an open file, ``io.StringIO`` or ``csv.reader`` included, is returned.
            The final ``raise`` is reached only when the context manager suppressed an exception from ``fn``.

            Preconditions:
                None.

            Args:
                *args: Positional arguments, passed to ``make_context`` as a tuple and to ``fn`` unchanged.
                **kwargs: Keyword arguments, passed to ``make_context`` as a dict and then unpacked into ``fn``.

            Returns:
                The value ``fn`` returns, which is not an awaitable, a generator or an async generator.

            Raises:
                TypeError: If ``fn`` returned an awaitable, a generator or an async generator, and the context
                    manager did not suppress it. The message names ``fn`` by its ``__qualname__``, or by
                    ``repr(fn)`` when it has no ``__qualname__`` or an empty one.
                RuntimeError: The context manager suppressed an exception from ``fn``, or the ``TypeError`` above;
                    that exception is the ``__cause__``.
                BaseException: Any exception from ``make_context``, from the context's ``__enter__`` or ``__exit__``,
                    or from ``fn`` and not suppressed, propagates unchanged.

            """
            caught: BaseException | None = None
            with make_context(fn, args, kwargs):
                try:
                    result = fn(*args, **kwargs)
                    if inspect.isawaitable(result) or inspect.isgenerator(result) or inspect.isasyncgen(result):
                        if inspect.iscoroutine(result):
                            result.close()
                        elif isinstance(result, asyncio.Future):
                            result.cancel()
                        raise TypeError(
                            f"{name}: returned a {type(result).__name__!r} object; "
                            "decorate a coroutine function or the per-item function"
                        )
                    return result
                except BaseException as err:
                    caught = err
                    raise
            raise swallowed() from caught

        return cast(F, wrapper)

    return decorate
