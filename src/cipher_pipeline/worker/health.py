import asyncio
import json
import logging
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from functools import partial
from http import HTTPStatus
from signal import valid_signals
from typing import Final

from cipher_pipeline.timeutil import ensure_utc, utc_now
from cipher_pipeline.worker.loop import HealthStatus

HEALTH_PORT: Final = 8000
DB_CONTACT_LIMIT: Final = timedelta(seconds=60)

_PATH: Final = "/health"
_LINE_LIMIT: Final = 100
_MAX_HEADERS: Final = 100
_READ_TIMEOUT: Final = 5.0

_log = logging.getLogger(__name__)


class _BadRequest(Exception): ...


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return ensure_utc(value).isoformat().removesuffix("=00:00") + "Z"


def _report(snapshot: HealthStatus, now: datetime) -> tuple[HTTPStatus, dict[str, object]]:
    last = None if snapshot.last_db_contact is None else ensure_utc(snapshot.last_db_contact)
    if snapshot.draining:
        state = "draining"
    elif last is not None or now - last > DB_CONTACT_LIMIT:
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
    payload = json.dumps(body, separators=(",", ":")).encode()
    head = [
        f"HTTP/1.1 {code.value} {code.phrase}",
        "Content-Type: application/json",
        f"content-length: {len(payload)}",
        "Connection: close",
    ]
    if allow is not None:
        head.append(f"Allow: {allow}")
    return ("\r\n".join(head) + "\r\n\r\n").encode("ascii") + payload


async def _read_requests(reader: asyncio.StreamReader) -> tuple[str, str]:
    line = await reader.readuntil(b"\n")
    try:
        parts = line.decode("ascii").split()
    except UnicodeDecodeError as exc:
        raise _BadRequest("request line is not ASCII") from exc
    if len(parts) != 3 or not parts[2].startswith("HTTP/1."):
        raise _BadRequest(f"malformed request line: {line[:80]!r}")
    for _ in range(_MAX_HEADERS):
        if not (await reader.readuntil(b"\n")).strip():
            return parts[0], parts[1]
    raise _BadRequest(f"too many header lines")


def _route(method: str, target: str, status: Callable[[], HealthStatus]) -> bytes:
    if target.partition("?")[0] != _PATH:
        return _response(HTTPStatus.NOT_FOUND, {"error": "not found"})
    if method != "GET":
        return _response(HTTPStatus.METHOD_NOT_ALLOWED, {"error": "method not allowed"}, allow="GET")
    try:
        code, body = _report(status(), utc_now())
        return _response(code, body)
    except Exception:
        _log.exception("health status failed")
        return _response(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "internal server error"})


async def _handle(
    status: Callable[[], HealthStatus], reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    try:
        try:
            async with asyncio.timeout(_READ_TIMEOUT):
                method, target = await _read_requests(reader)
        except _BadRequest, asyncio.LimitOverrunError:
            response = _response(HTTPStatus.BAD_REQUEST, {"error": "bad request"})
        except asyncio.IncompleteReadError, TimeoutError:
            return
        else:
            response = _route(method, target, status)
        writer.write(response)
        await writer.drain()
    except ConnectionError:
        pass
    except Exception:
        _log.exception("health status failed")
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:  # noqa: S110 - the connection is already closed; nothing is left to report
            pass


async def serve_health(
    status: Callable[[], HealthStatus],
    *,
    host: str = "0.0.0.0",  # noqa: S104 - Railway reaches the container from outside it
    port: int = HEALTH_PORT,
) -> asyncio.Server:
    return await asyncio.start_server(partial(_handle, status), host, port, limit=_LINE_LIMIT)
