"""The worker's entry point: ``python -m pipeline``.

Start-up runs in a fixed order and any failure raises, so the process exits non-zero before ``/health`` answers: a bad
deploy fails the host's health check and the previous deployment keeps running. Shutdown runs in reverse and needs
25 seconds between SIGTERM and SIGKILL: 15 for the drain and 10 for telemetry. The host's stop grace period must be
at least that; the deployment config sets it.

Public API:
    main: Read the settings, pin Polars, set up telemetry and run the worker until it is stopped.
    run_worker: Open the pool, check the schema, run the worker and ``/health``, then shut down in order.
"""

import asyncio
import contextlib
import logging
import signal
from collections.abc import Mapping
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


async def _close_server(server: asyncio.Server) -> None:
    """Stop the ``/health`` server and wait until it has closed.

    Preconditions:
        None.

    Args:
        server: The server ``serve_health`` returned.

    Returns:
        None, once the listening socket is closed and every open connection has ended, which takes at most
        ``_READ_TIMEOUT`` (5 s) because ``serve_health`` closes each connection after one response.

    Raises:
        Nothing.

    """
    server.close()
    await server.wait_closed()


def _remove_signal_handlers(loop: asyncio.AbstractEventLoop) -> None:
    """Remove the SIGTERM and SIGINT handlers, if any were added.

    Preconditions:
        Called on ``loop``'s thread.

    Args:
        loop: The loop the handlers were added to.

    Returns:
        None. A signal with no handler is skipped.

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
    then add SIGTERM and SIGINT handlers that call ``Worker.stop``. Shutdown, once ``Worker.run`` has drained or
    anything raised: close ``/health``, close the listener, close the pool, remove the handlers. Each shutdown step
    runs even if an earlier one raised, and only the steps whose resource was opened run. A signal that arrives before
    the handlers are added has its default effect and ends the process at once; no task has been claimed by then, and
    the closed sockets end its database sessions.

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
        None, once the worker has been stopped, has drained and everything is closed.

    Raises:
        DatabaseUnavailable: The pool or the listener could not reach the warehouse.
        SchemaMismatch: The warehouse is below ``REQUIRED_REVISION``, or its revision cannot be read.
        OSError: ``/health`` could not bind its port.
        Exception: Anything else start-up or shutdown raises, unchanged.

    """
    loop = asyncio.get_running_loop()
    async with contextlib.AsyncExitStack() as shutdown:  # unwinds in reverse; every step runs even if one raises
        shutdown.callback(_remove_signal_handlers, loop)  # last: a second signal while closing only repeats stop()
        pool = await open_pool(settings)
        shutdown.push_async_callback(pool.close)
        await check_schema(pool)
        listener = await open_listener(settings)
        shutdown.push_async_callback(listener.close)
        worker = Worker(
            settings=settings,
            pool=pool,
            listener=listener,
            runners=runners,
            known_jobs=known_jobs,
            instruments=instruments,
        )
        server = await serve_health(worker.health, port=health_port)
        shutdown.push_async_callback(_close_server, server)
        for sig in _SIGNALS:
            loop.add_signal_handler(sig, worker.stop)
        _log.info("worker %s started with %d slots", settings.worker_id, settings.concurrency)
        await worker.run()
        _log.info("worker %s drained; shutting down", settings.worker_id)


def main() -> None:
    """Read the settings, pin Polars, set up telemetry and run the worker until it is stopped.

    Until the job registry loader exists the worker knows no jobs: any claimed task fails as ``UnknownJob``.

    Preconditions:
        Called once per process, before anything imports Polars or job code.

    Returns:
        None, once the worker has stopped and telemetry has been flushed.

    Raises:
        pydantic.ValidationError: The settings are missing or invalid.
        StartupRefused: A compute resource check failed.
        RuntimeError: Telemetry could not be configured, for example no OTLP endpoint outside ``local``.
        Exception: Whatever ``run_worker`` raises, after it is logged and telemetry has been shut down.
            ``Telemetry.shutdown`` logs each failed step itself and raises no ``Exception``, so it never replaces
            this one.

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
