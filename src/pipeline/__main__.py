"""The worker's entry point: ``python -m pipeline``.

Start-up runs in a fixed order and any failure raises, so the process exits non-zero before ``/health`` answers: a bad
deploy fails the host's health check and the previous deployment keeps running. Shutdown runs in reverse and budgets
30 seconds between SIGTERM and SIGKILL: 15 (``SHUTDOWN_TIMEOUT``) from the stop signal for everything before telemetry
(the drain, whose grace period is ``DRAIN_TIMEOUT``'s 10, then its cut-off and the failure reports it writes, then
closing ``/health``, the listener and the pool), 10 for telemetry (``Telemetry.shutdown``) and 5 spare. Whatever is
still running when ``SHUTDOWN_TIMEOUT`` runs out is cancelled and logged (see ``run_worker``). The host's stop grace
period must be at least the budget; the deployment config, which is not in this repository, sets it.

Public API:
    SHUTDOWN_TIMEOUT: Longest time from the stop signal, or the start of shutdown, to the end of ``run_worker``.
    main: Read the settings, pin Polars, set up telemetry and run the worker until it is stopped.
    run_worker: Open the pool, check the schema, run the worker and ``/health``, then shut down in order.
"""

import asyncio
import contextlib
import logging
import signal
from collections.abc import Awaitable, Callable, Mapping
from datetime import timedelta
from typing import Final

from pipeline.config import Settings
from pipeline.db.control import open_listener
from pipeline.db.pool import check_schema, open_pool
from pipeline.telemetry import Instruments, configure
from pipeline.worker.health import HEALTH_PORT, serve_health
from pipeline.worker.loop import TaskRunner, Worker
from pipeline.worker.resources import pin_polars_threads

_SIGNALS: Final = (signal.SIGTERM, signal.SIGINT)  # the host sends SIGTERM to stop; SIGINT is Ctrl-C locally

_log: Final = logging.getLogger(__name__)

#: Longest time from the stop signal (or from the start of shutdown, without one) to the end of ``run_worker``: the
#: drain and its cut-off after ``DRAIN_TIMEOUT``'s 10 s, the cut-off tasks' reports and the closes of ``/health``, the
#: listener and the pool. ``Telemetry.shutdown``'s 10 s follow it, leaving 5 s of the 30 s budget spare.
SHUTDOWN_TIMEOUT: Final = timedelta(seconds=15)


class _Deadline:
    """The shutdown deadline: none until ``start``, then ``timeout`` after the first ``start``.

    ``run`` runs one shutdown step, abandoning it if the deadline passes first; once it has passed, every later step
    is skipped. ``close`` starts the deadline, if nothing has yet, and then runs a step. Created and used on one event
    loop.

    Attributes:
        None public. All state is private.

    """

    def __init__(self, loop: asyncio.AbstractEventLoop, timeout: timedelta) -> None:
        """Create a deadline that has not started.

        Preconditions:
            None.

        Args:
            loop: The running loop the steps run on.
            timeout: Time from ``start`` to the deadline.

        Returns:
            None.

        Raises:
            Nothing.

        """
        self._loop = loop
        self._timeout = timeout
        self._passed = asyncio.Event()
        self._timer: asyncio.TimerHandle | None = None

    def start(self) -> None:
        """Start the deadline ``timeout`` from now, unless it has already started.

        Preconditions:
            Called on the loop's thread.

        Returns:
            None.

        Raises:
            Nothing.

        """
        if self._timer is None:
            self._timer = self._loop.call_later(self._timeout.total_seconds(), self._passed.set)

    def cancel(self) -> None:
        """Cancel the timer, if one is set, so it does not fire after the loop has moved on.

        Preconditions:
            Called on the loop's thread.

        Returns:
            None.

        Raises:
            Nothing.

        """
        if self._timer is not None:
            self._timer.cancel()

    async def close(self, what: str, step: Callable[[], Awaitable[None]]) -> bool:
        """Start the deadline, unless it has already started, then run ``step`` as ``run`` does.

        Preconditions:
            Awaited on the deadline's loop.

        Args:
            what: What the step does, for the log, e.g. ``"closing the pool"``.
            step: Returns the step's awaitable.

        Returns:
            What ``run`` returns.

        Raises:
            Exception: Whatever ``run`` raises.
            asyncio.CancelledError: If the call is cancelled.

        """
        self.start()
        return await self.run(what, step)

    async def run(self, what: str, step: Callable[[], Awaitable[None]]) -> bool:
        """Run ``step`` until it ends or the deadline passes, whichever comes first.

        At the deadline, ``step`` is cancelled and not awaited, which is logged at ``ERROR``. A step called after the
        deadline has passed is not started, and that is logged at ``ERROR`` too.

        Preconditions:
            Awaited on the deadline's loop.

        Args:
            what: What the step does, for the log, e.g. ``"closing the pool"``.
            step: Returns the step's awaitable; it is called only if the deadline has not passed.

        Returns:
            ``True`` once ``step`` has ended before the deadline; ``False`` if it was abandoned or skipped.

        Raises:
            Exception: Whatever ``step`` raises, unchanged, when it ends before the deadline.
            asyncio.CancelledError: If the call is cancelled; ``step`` is cancelled too.

        """
        if self._passed.is_set():
            _log.error("shutdown deadline passed; %s skipped", what)
            return False
        work = asyncio.ensure_future(step())
        passed = asyncio.create_task(self._passed.wait())
        try:
            await asyncio.wait([work, passed], return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            work.cancel()
            raise
        finally:
            passed.cancel()
        if work.done():
            work.result()
            return True
        work.cancel()
        _log.error(
            "shutdown did not finish within %g s of the stop signal; %s abandoned",
            self._timeout.total_seconds(),
            what,
        )
        return False


async def _close_server(server: asyncio.Server) -> None:
    """Stop the ``/health`` server, close its open connections and wait until every connection has been dropped.

    ``close`` stops accepting connections and ``close_clients`` closes every open one, so a client that has connected
    but not yet sent its request cannot hold this call for ``_READ_TIMEOUT``; ``wait_closed`` then waits until every
    connection's transport has been dropped. It does not wait for the connection handlers' tasks: a handler can still
    be running when this returns, and one still pending when ``asyncio.run`` ends is cancelled there.

    Preconditions:
        Awaited on the loop that started ``server``.

    Args:
        server: The server ``serve_health`` returned.

    Returns:
        None, once the listening socket is closed and every open connection has been dropped.

    Raises:
        asyncio.CancelledError: If the call is cancelled.

    """
    server.close()
    server.close_clients()
    await server.wait_closed()


def _stop(worker: Worker, deadline: _Deadline) -> None:
    """Handle a stop signal: ask the worker to stop and start the shutdown deadline.

    Preconditions:
        Called on the loop's thread, as a signal handler ``loop.add_signal_handler`` added.

    Args:
        worker: The worker to stop.
        deadline: The shutdown deadline; a later signal does not move it.

    Returns:
        None.

    Raises:
        Nothing.

    """
    worker.stop()
    deadline.start()


def _remove_signal_handlers(loop: asyncio.AbstractEventLoop) -> None:
    """Remove the SIGTERM and SIGINT handlers, if any were added.

    ``remove_signal_handler`` restores SIGTERM to its default action and SIGINT to ``signal.default_int_handler``, so a
    later SIGINT raises ``KeyboardInterrupt``.

    Preconditions:
        Called on the main thread, on a Unix loop (``loop.remove_signal_handler`` is not available elsewhere).

    Args:
        loop: The loop the handlers were added to.

    Returns:
        None. A signal with no handler on ``loop`` is a no-op.

    Raises:
        Nothing.

    """
    for sig in _SIGNALS:
        loop.remove_signal_handler(sig)


async def run_worker(
    settings: Settings,
    instruments: Instruments,
    *,
    runners: Mapping[str, TaskRunner],
    known_jobs: frozenset[str],
    health_port: int = HEALTH_PORT,
) -> None:
    """Open the pool, check the schema, run the worker and ``/health`` until stopped, then shut down in order.

    Start-up: open the pool, check the schema revision, open the listener, build the ``Worker``, start ``/health``,
    then add SIGTERM and SIGINT handlers that call ``Worker.stop`` and start the shutdown deadline. No ``await``
    separates adding the handlers from starting ``Worker.run``. Shutdown, once ``Worker.run`` has drained or anything
    raised: close ``/health``, close the listener, close the pool, remove the handlers. Each shutdown step runs even if
    an earlier one raised, and only the steps whose resource was opened run; the handler removal always runs.
    ``/health`` stays up during the drain, so it reports the worker as draining. A SIGTERM before the handlers are
    added has its default effect and ends the process at once; no task has been claimed by then, and the closed
    sockets end its database sessions. A SIGINT before then reaches ``asyncio.run``'s own handler, which cancels this
    coroutine, so shutdown unwinds as for any other raise. A signal after the handlers are added only calls
    ``Worker.stop`` again, so a second SIGTERM or Ctrl-C cannot cut the drain short or move the deadline.

    The first stop signal starts one deadline, ``SHUTDOWN_TIMEOUT`` later, over ``Worker.run`` (its drain, the
    cut-off after ``DRAIN_TIMEOUT`` and the cut-off tasks' reports) and the three closes. Without a stop signal, for
    example when start-up or ``Worker.run`` raises, the first close starts it, so the closes are always bounded. If it
    passes, the step then running is cancelled without being awaited and logged at ``ERROR`` (a cancelled
    ``Worker.run`` cancels its running task slots), the remaining closes are skipped and logged, and this coroutine
    returns.

    Preconditions:
        Awaited once, inside a running Unix event loop on the main thread (signal handlers can only be added and
        removed there).

    Args:
        settings: The worker's settings.
        instruments: Where the worker records its metrics.
        runners: The runner for each job that can run.
        known_jobs: Every job in the job registry.
        health_port: The port ``/health`` listens on; tests pass a free one.

    Returns:
        None, once the worker has been stopped, has drained and everything is closed, or once the shutdown deadline
        has passed.

    Raises:
        DatabaseUnavailable: The pool or the listener could not reach the warehouse (as mapped by ``db_errors``).
        SchemaMismatch: ``alembic_version`` holds no single valid revision, or one below ``REQUIRED_REVISION``.
        OSError: ``/health`` could not bind its port.
        asyncio.CancelledError: The coroutine was cancelled, for example by a SIGINT before the handlers were added.
        Exception: Anything else start-up, ``Worker.run`` or shutdown raises, unchanged, such as a ``db_errors``
            mapping of a missing ``alembic_version`` table or ``DatabaseConflict`` from the listener's ``LISTEN``. If
            a shutdown step raises after an earlier error, its exception propagates in place of the earlier one, which
            becomes its ``__context__``. A step abandoned at the deadline raises nothing.

    """
    loop = asyncio.get_running_loop()
    deadline = _Deadline(loop, SHUTDOWN_TIMEOUT)
    async with contextlib.AsyncExitStack() as shutdown:  # unwinds in reverse; every step runs even if one raises
        shutdown.callback(deadline.cancel)
        shutdown.callback(_remove_signal_handlers, loop)  # last: a second signal while closing only repeats stop()
        pool = await open_pool(settings)
        shutdown.push_async_callback(deadline.close, "closing the pool", pool.close)
        await check_schema(pool)
        listener = await open_listener(settings)
        shutdown.push_async_callback(deadline.close, "closing the listener", listener.close)
        worker = Worker(
            settings=settings,
            pool=pool,
            listener=listener,
            runners=runners,
            known_jobs=known_jobs,
            instruments=instruments,
        )
        server = await serve_health(worker.health, port=health_port)
        shutdown.push_async_callback(deadline.close, "closing /health", lambda: _close_server(server))
        for sig in _SIGNALS:
            loop.add_signal_handler(sig, _stop, worker, deadline)
        _log.info("worker %s started with %d slots", settings.worker_id, settings.concurrency)
        if await deadline.run("draining the worker", worker.run):
            _log.info("worker %s drained; shutting down", settings.worker_id)


def main() -> None:
    """Read the settings, pin Polars, set up telemetry and run the worker until it is stopped.

    Until the job registry loader exists the worker knows no jobs: any claimed task fails as ``UnknownJob``. Settings
    and Polars failures raise before logging is configured, so they reach stderr only as the uncaught traceback. None
    of this module's imports loads Polars, so ``pin_polars_threads`` runs before the first import of it.

    Preconditions:
        Called once per process, on the main thread, before anything imports Polars or job code.

    Returns:
        None, once the worker has stopped and telemetry has been flushed.

    Raises:
        pydantic.ValidationError: The settings are missing or invalid.
        StartupRefused: A compute resource check failed.
        ImportError: Polars could not be imported.
        UnicodeDecodeError: A cgroup file read by ``pin_polars_threads`` is not valid text.
        RuntimeError: Telemetry could not be configured, for example no OTLP endpoint outside ``local``.
        BaseException: Anything ``configure`` raises while registering providers, instrumenting or attaching
            handlers, after it has shut its own telemetry down. ``log_level`` is a ``Literal`` on ``Settings``, so
            ``configure``'s ``ValueError`` for an unknown level cannot arise here.
        Exception: Whatever ``run_worker`` raises, after it is logged at ``CRITICAL`` and telemetry has been shut
            down. ``Telemetry.shutdown`` logs each failed step itself and raises no ``Exception``, so it never
            replaces this one.
        KeyboardInterrupt: A SIGINT while ``asyncio.run`` runs, before the handlers were added or after they were
            removed; it is not logged, and telemetry is still shut down. A SIGINT before ``asyncio.run`` starts
            raises it from settings, Polars or telemetry set-up, before the ``try``, so telemetry is not shut down.

    """
    settings = Settings()
    pin_polars_threads(settings.polars_threads)
    telemetry = configure(env=settings.env, worker_id=settings.worker_id, log_level=settings.log_level)
    try:
        asyncio.run(run_worker(settings, telemetry.instruments, runners={}, known_jobs=frozenset()))
    except Exception:
        _log.critical("worker %s stopped on an error", settings.worker_id, exc_info=True)
        raise
    finally:
        telemetry.shutdown()


if __name__ == "__main__":
    main()
