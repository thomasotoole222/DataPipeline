"""The worker's HTTP health endpoint: answer ``GET /health`` (and ``HEAD /health``) with a JSON snapshot of the worker.

Public API:
    serve_health: Start the health server.
    HEALTH_PORT: Default port ``serve_health`` listens on.
    DB_CONTACT_LIMIT: Longest gap since the last database contact that the report still calls ``"ok"``.

"""

import asyncio
import contextlib
import json
import logging
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from functools import partial
from http import HTTPStatus
from typing import Final

from pipeline.timeutil import utc_now
from pipeline.worker.loop import HealthStatus

#: Default port the health server listens on.
HEALTH_PORT: Final = 8080
#: Longest time since the last database contact that still counts as healthy; a gap of exactly this long is healthy.
DB_CONTACT_LIMIT: Final = timedelta(seconds=60)

#: The only path the server answers; a query string is ignored.
_PATH: Final = "/health"
#: Stream buffer limit in bytes; a request or header line that does not fit is answered with 400.
_LINE_LIMIT: Final = 8192
#: Most lines read after the request line, the blank line that ends the headers included.
_MAX_HEADERS: Final = 100
#: Seconds allowed to read the request line and headers before the connection is closed without a response.
_READ_TIMEOUT: Final = 5.0
#: Most seconds spent discarding unread input after a 400, so a client still sending cannot hold the handler.
_DISCARD_TIMEOUT: Final = 1.0

#: Logger for this module.
_log = logging.getLogger(__name__)


class _BadRequest(Exception):
    """Signal a malformed request, which ``_handle`` answers with 400 Bad Request."""

    ...


def _iso(value: datetime | None) -> str | None:
    """Format ``value`` as an ISO 8601 UTC timestamp with a ``Z`` suffix.

    ``value`` is formatted with ``isoformat``, then the ``"+00:00"`` offset is replaced with ``"Z"``. Its only caller,
    ``_report``, passes ``HealthStatus`` fields, which ``HealthStatus`` stores in UTC.

    Preconditions:
        ``value`` is a ``datetime`` with ``tzinfo`` set to ``UTC``, or ``None``.

    Args:
        value: The time to format.

    Returns:
        ``None`` if ``value`` is ``None``; otherwise the formatted text, e.g. ``"2026-10-14T12:00:00Z"``.

    Raises:
        Nothing.

    """
    if value is None:
        return None
    return value.isoformat().removesuffix("+00:00") + "Z"


def _report(snapshot: HealthStatus, now: datetime) -> tuple[HTTPStatus, dict[str, object]]:
    """Build the HTTP status and JSON body for ``snapshot``.

    The state is ``"draining"`` if the worker is draining, otherwise ``"db_unreachable"`` if there has been no
    database contact or the last one was more than ``DB_CONTACT_LIMIT`` before ``now``, otherwise ``"ok"``.
    The body holds ``worker_id``, ``status``, ``busy``, ``slots``, ``draining``, and ``paused_until`` and
    ``last_db_contact`` formatted by ``_iso``.

    ``now - last_db_contact`` is the age of the last contact as the worker measured it on its monotonic clock, because
    ``Worker.health`` places ``last_db_contact`` that far before its own ``utc_now()``; a step of the system clock
    therefore cannot hide or fake a lost database.

    Preconditions:
        ``now`` is the current time as an aware ``datetime``, read right after ``snapshot`` was taken.

    Args:
        snapshot: The worker's health snapshot.
        now: The time to judge the last database contact against.

    Returns:
        ``(HTTPStatus.OK, body)`` when the state is ``"ok"``, otherwise ``(HTTPStatus.SERVICE_UNAVAILABLE, body)``.

    Raises:
        Nothing: ``HealthStatus.__post_init__`` stores ``last_db_contact`` and ``paused_until`` in UTC, as ``_iso``
            requires.

    """
    last = snapshot.last_db_contact
    if snapshot.draining:
        state = "draining"
    elif last is None or now - last > DB_CONTACT_LIMIT:
        state = "db_unreachable"
    else:
        state = "ok"
    body: dict[str, object] = {
        "worker_id": snapshot.worker_id,
        "status": state,
        "busy": snapshot.busy,
        "slots": snapshot.slots,
        "draining": snapshot.draining,
        "paused_until": _iso(snapshot.paused_until),
        "last_db_contact": _iso(last),
    }
    return (HTTPStatus.OK if state == "ok" else HTTPStatus.SERVICE_UNAVAILABLE), body


def _response(code: HTTPStatus, body: Mapping[str, object], *, allow: str | None = None) -> bytes:
    """Encode a complete HTTP/1.1 response with a compact JSON body.

    The head has the status line, ``Content-Type: application/json``, ``Content-Length``, ``Connection: close``
    and, when ``allow`` is given, an ``Allow`` header. The head and body are ASCII: ``json.dumps`` escapes non-ASCII.

    Preconditions:
        None.

    Args:
        code: The response status.
        body: The JSON body.
        allow: Value of the ``Allow`` header, or ``None`` to leave it out.

    Returns:
        The head, a blank line and the JSON body, as bytes.

    Raises:
        TypeError: If a value in ``body`` is not JSON serialisable (raised by ``json.dumps``).
        ValueError: If ``body`` contains a circular reference (raised by ``json.dumps``).
        UnicodeEncodeError: If ``allow`` is not ASCII.

    """
    payload = json.dumps(body, separators=(",", ":")).encode()
    head = [
        f"HTTP/1.1 {code.value} {code.phrase}",
        "Content-Type: application/json",
        f"Content-Length: {len(payload)}",
        "Connection: close",
    ]
    if allow is not None:
        head.append(f"Allow: {allow}")
    return ("\r\n".join(head) + "\r\n\r\n").encode("ascii") + payload


async def _read_request(reader: asyncio.StreamReader) -> tuple[str, str]:
    """Read one request's request line and headers, and return its method and target.

    The request line must be ASCII and split on whitespace into three parts, the last starting with ``"HTTP/1."``.
    Lines are then read until a blank one, up to ``_MAX_HEADERS`` lines; their contents are ignored. A request body
    is not read.

    Preconditions:
        ``reader`` is at the start of a request.

    Args:
        reader: The connection's stream reader.

    Returns:
        ``(method, target)`` from the request line, e.g. ``("GET", "/health")``.

    Raises:
        _BadRequest: If the request line is not ASCII or is malformed, or no blank line comes within
            ``_MAX_HEADERS`` lines.
        asyncio.IncompleteReadError: If the connection closes before a line ends.
        asyncio.LimitOverrunError: If a line does not fit in the stream's buffer limit.
        ConnectionError: If the transport fails while reading, e.g. the peer resets the connection.
        asyncio.CancelledError: If the call is cancelled.

    """
    line = await reader.readuntil(b"\n")
    try:
        parts = line.decode("ascii").split()
    except UnicodeDecodeError as err:
        raise _BadRequest("request line is not ASCII") from err
    if len(parts) != 3 or not parts[2].startswith("HTTP/1."):
        raise _BadRequest(f"malformed request line: {line[:80]!r}")
    for _ in range(_MAX_HEADERS):
        if not (await reader.readuntil(b"\n")).strip():
            return parts[0], parts[1]
    raise _BadRequest("too many header lines")


def _route(method: str, target: str, status: Callable[[], HealthStatus]) -> bytes:
    """Return the response for a request with ``method`` and ``target``.

    A path other than ``_PATH``, ignoring any query string, gets 404 Not Found. A method other than ``GET`` or
    ``HEAD`` gets 405 Method Not Allowed with ``Allow: GET, HEAD``. A ``GET`` gets the response ``_report`` builds from
    ``status()`` and ``utc_now()``. A ``HEAD`` gets the response the same ``GET`` would get, with the same status and
    headers (``Content-Length`` included) but no body.

    Preconditions:
        None.

    Args:
        method: The request method.
        target: The request target.
        status: Returns the worker's current health snapshot.

    Returns:
        The encoded response.

    Raises:
        Nothing: an ``Exception`` from ``status``, ``_report`` or encoding the report is logged at ``ERROR`` with
            its traceback and answered with 500 Internal Server Error.

    """
    response = _route_get(method, target, status)
    if method == "HEAD":
        head, _, _ = response.partition(b"\r\n\r\n")
        return head + b"\r\n\r\n"
    return response


def _route_get(method: str, target: str, status: Callable[[], HealthStatus]) -> bytes:
    """Return the full response, body included, for a request with ``method`` and ``target``, ``HEAD`` as ``GET``.

    Preconditions:
        None.

    Args:
        method: The request method.
        target: The request target.
        status: Returns the worker's current health snapshot.

    Returns:
        404 Not Found for a path other than ``_PATH`` (ignoring any query string); 405 Method Not Allowed with
        ``Allow: GET, HEAD`` for a method other than ``GET`` or ``HEAD``; otherwise the response ``_report`` builds
        from ``status()`` and ``utc_now()``. Every response carries a JSON body.

    Raises:
        Nothing: an ``Exception`` from ``status``, ``_report`` or encoding the report is logged at ``ERROR`` with
            its traceback and answered with 500 Internal Server Error.

    """
    if target.partition("?")[0] != _PATH:
        return _response(HTTPStatus.NOT_FOUND, {"error": "not found"})
    if method not in {"GET", "HEAD"}:
        return _response(HTTPStatus.METHOD_NOT_ALLOWED, {"error": "method not allowed"}, allow="GET, HEAD")
    try:
        code, body = _report(status(), utc_now())
        return _response(code, body)
    except Exception:
        _log.exception("health status failed")
        return _response(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "status failed"})


async def _discard(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Half-close the write side, then discard unread input until end of input or ``_DISCARD_TIMEOUT`` seconds.

    Closing a connection with unread input pending resets it, which can lose a response the client has not read
    yet. Sending end of output first and reading what is left lets the response reach a client that is still
    sending. The wait is bounded, so a client that keeps sending only holds the handler for ``_DISCARD_TIMEOUT``.

    Preconditions:
        The response has been written and drained; nothing else is written to ``writer`` afterwards.

    Args:
        reader: The connection's stream reader.
        writer: The connection's stream writer.

    Returns:
        None, at end of input or once ``_DISCARD_TIMEOUT`` has passed, whichever comes first.

    Raises:
        ConnectionError: If the transport fails while half-closing or reading, e.g. the peer resets the connection.
        asyncio.CancelledError: If the call is cancelled.

    """
    writer.write_eof()
    with contextlib.suppress(TimeoutError):
        async with asyncio.timeout(_DISCARD_TIMEOUT):
            while await reader.read(_LINE_LIMIT):
                pass


async def _handle(
    status: Callable[[], HealthStatus], reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    """Serve one request on a connection, then close it.

    The request line and headers must arrive within ``_READ_TIMEOUT`` seconds. A malformed request or a line that
    overruns the buffer limit is answered with 400 Bad Request. If the connection closes early or the timeout
    expires, it is closed without a response. Otherwise the response comes from ``_route``. The timeout does not
    cover writing the response.

    A ``ConnectionError`` while reading or writing is swallowed; any other ``Exception`` is logged at ``ERROR`` with
    its traceback and swallowed. The writer is always closed, and an ``Exception`` while waiting for the close is
    swallowed.

    After a 400, ``_discard`` half-closes the connection and discards unread input for up to ``_DISCARD_TIMEOUT``
    seconds before the close, so the 400 reaches a client that is still sending (e.g. a 1 MB request line) rather
    than being lost to a reset. After any other response the connection is closed at once.

    Preconditions:
        Called by the server ``serve_health`` starts, once per connection.

    Args:
        status: Returns the worker's current health snapshot.
        reader: The connection's stream reader.
        writer: The connection's stream writer.

    Returns:
        None, after the connection is closed.

    Raises:
        asyncio.CancelledError: If the call is cancelled; the connection is still closed.

    """
    try:
        try:
            async with asyncio.timeout(_READ_TIMEOUT):
                method, target = await _read_request(reader)
        except _BadRequest, asyncio.LimitOverrunError:
            writer.write(_response(HTTPStatus.BAD_REQUEST, {"error": "bad request"}))
            await writer.drain()
            await _discard(reader, writer)
            return
        except asyncio.IncompleteReadError, TimeoutError:
            return
        writer.write(_route(method, target, status))
        await writer.drain()
    except ConnectionError:
        pass
    except Exception:
        _log.exception("health request failed")
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:  # noqa: S110 - the connection is already closed; nothing is left to report
            pass


async def serve_health(
    status: Callable[[], HealthStatus],
    *,
    host: str = "0.0.0.0",  # noqa: S104 - the host reaches the container from outside it
    port: int = HEALTH_PORT,
) -> asyncio.Server:
    """Start an HTTP server that answers ``GET /health`` with the worker's health, and ``HEAD /health`` with its head.

    Each connection serves one request through ``_handle`` and is then closed. The streams' buffer limit is
    ``_LINE_LIMIT`` bytes. The server is already accepting connections when it is returned; the caller closes it
    with ``close`` and ``wait_closed``.

    Preconditions:
        Awaited from a running event loop. ``status`` is synchronous and safe to call on the event loop; it is
        called once per ``GET`` or ``HEAD`` of ``/health``.

    Args:
        status: Returns the worker's current health snapshot, e.g. ``Worker.health``.
        host: Address to listen on.
        port: Port to listen on.

    Returns:
        The running server.

    Raises:
        OSError: If the server cannot listen on ``host`` and ``port``, e.g. because the port is in use.
        asyncio.CancelledError: If the call is cancelled.

    """
    return await asyncio.start_server(partial(_handle, status), host, port, limit=_LINE_LIMIT)
