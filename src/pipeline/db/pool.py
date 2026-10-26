"""The warehouse connection pool and the schema revision check run at startup.

Public API:
    Connection: An async psycopg connection that returns tuple rows.
    Pool: An async connection pool of ``Connection`` objects.
    REQUIRED_REVISION: The lowest database schema revision this build accepts.
    DEAD_CONNECTION_PARAMS: libpq parameters that detect a dead connection.
    open_pool: Open the warehouse connection pool and wait until it is ready.
    check_schema: Check that the database schema revision meets ``REQUIRED_REVISION``.
"""

from collections.abc import Mapping
from datetime import timedelta
from typing import Final

from psycopg import AsyncConnection
from psycopg.rows import TupleRow
from psycopg_pool import AsyncConnectionPool

from pipeline.config import Settings
from pipeline.errors import SchemaMismatch, db_errors

type Connection = AsyncConnection[TupleRow]
type Pool = AsyncConnectionPool[Connection]

#: Lowest ``alembic_version`` revision of the database schema that this build accepts.
REQUIRED_REVISION: Final = 0

#: Longest wait for the pool to open, and for a connection to be checked out of it.
_POOL_TIMEOUT: Final = timedelta(seconds=10)
#: Server-side ``statement_timeout`` set on every pool connection.
_STATEMENT_TIMEOUT: Final = timedelta(seconds=60)
#: Server-side ``idle_in_transaction_session_timeout`` set on every pool connection.
_IDLE_IN_TRANSACTION_TIMEOUT: Final = timedelta(seconds=60)
#: Idle time on a connection before the first TCP keepalive probe is sent.
_KEEPALIVES_IDLE: Final = timedelta(seconds=60)  # idle time before the first TCP keepalive probe
#: Time between unanswered TCP keepalive probes.
_KEEPALIVES_INTERVAL: Final = timedelta(seconds=15)  # time between unanswered keepalive probes
#: Unanswered TCP keepalive probes after which the connection is treated as dead.
_KEEPALIVES_COUNT: Final = 4  # unanswered keepalive probes before the connection is dead
#: Longest time sent data may stay unacknowledged before the connection is treated as dead.
_TCP_USER_TIMEOUT: Final = timedelta(seconds=120)  # time sent data may go unacknowledged before the connection is dead


def _ms(value: timedelta) -> int:
    """Convert a duration to whole milliseconds.

    Divides ``value`` by one millisecond with ``timedelta`` floor division, which is exact integer arithmetic on
    microseconds, so there is no floating-point rounding error at any size.

    Preconditions:
        None.

    Args:
        value: The duration to convert.

    Returns:
        ``value // timedelta(milliseconds=1)``: the exact number of whole milliseconds in ``value``, truncated so any
        sub-millisecond remainder is dropped. Floor division rounds towards negative infinity, so a negative duration
        with a remainder rounds down (``timedelta(microseconds=-1500)`` gives ``-2``).

    Raises:
        None.

    """
    return value // timedelta(milliseconds=1)


# libpq parameters that detect a dead connection in about 2 minutes; db.control.open_listener uses them too
#: libpq parameters that detect a dead connection in about 2 minutes, used by the pool and by
#: ``db.control.open_listener``.
DEAD_CONNECTION_PARAMS: Final[Mapping[str, int]] = {
    "keepalives": 1,
    "keepalives_idle": _KEEPALIVES_IDLE // timedelta(seconds=1),
    "keepalives_interval": _KEEPALIVES_INTERVAL // timedelta(seconds=1),
    "keepalives_count": _KEEPALIVES_COUNT,
    "tcp_user_timeout": _ms(_TCP_USER_TIMEOUT),
}


async def open_pool(settings: Settings) -> Pool:
    """Open the warehouse connection pool and wait until it is ready.

    Every connection the pool creates uses autocommit, takes ``settings.worker_id`` as its ``application_name``, and
    sets ``statement_timeout`` and ``idle_in_transaction_session_timeout`` through the ``options`` connection
    parameter.
    Every connection also takes the libpq parameters in DEAD_CONNECTION_PARAMS (TCP keepalives and tcp_user_timeout),
    so a connection whose server has gone is found dead in about 2 minutes.

    If ``pool.open`` fails or is interrupted, the pool is closed before the exception propagates.


    Preconditions:
        Must be awaited inside a running event loop. ``settings.concurrency`` must be an int of at least 1.
        ``settings.warehouse_url`` must hold a connection string that ``AsyncConnectionPool`` accepts.

    Args:
        settings: Supplies the connection string (``warehouse_url``), the pool's maximum size
            (``2 * concurrency``) and each connection's ``application_name`` (``worker_id``).

    Returns:
        The opened pool, named ``"warehouse"``, with ``min_size`` 1, ``max_size`` ``2 * settings.concurrency`` and a
        checkout timeout of ``_POOL_TIMEOUT``.

    Raises:
        Exception: Any error from the ``AsyncConnectionPool`` constructor, unchanged, since it runs outside
            ``db_errors``.
        Exception: Any error from ``pool.open``, including its timeout after ``_POOL_TIMEOUT``, as translated by
            ``db_errors``.
        BaseException: Any exception from ``pool.open``, ``asyncio.CancelledError`` included, is re-raised after
            ``pool.close`` has been awaited. Nothing is swallowed.

    """
    options = (
        f"-c statement_timeout={_ms(_STATEMENT_TIMEOUT)} "
        f"-c idle_in_transaction_session_timeout={_ms(_IDLE_IN_TRANSACTION_TIMEOUT)}"
    )
    pool = AsyncConnectionPool(
        settings.warehouse_url.get_secret_value(),
        min_size=1,
        max_size=2 * settings.concurrency,
        timeout=_POOL_TIMEOUT.total_seconds(),
        kwargs={
            "autocommit": True,
            "application_name": settings.worker_id,
            "options": options,
            **DEAD_CONNECTION_PARAMS,
        },
        check=AsyncConnectionPool.check_connection,
        name="warehouse",
        open=False,
    )
    try:
        with db_errors():
            await pool.open(wait=True, timeout=_POOL_TIMEOUT.total_seconds())
    except BaseException:
        await pool.close()
        raise
    return pool


async def check_schema(pool: Pool) -> int:
    """Check that the database schema revision meets ``REQUIRED_REVISION``.

    Reads every row of the ``alembic_version`` table through a connection taken from ``pool``, inside ``db_errors``.
    The connection is released before any validation runs. The table must hold exactly one row whose
    ``version_num`` is a ``str`` made only of decimal digits. That string is converted with ``int`` and compared
    with ``REQUIRED_REVISION``.

    Preconditions:
        ``pool`` must be open and able to hand out a connection. The connected database must contain an
        ``alembic_version`` table with a ``version_num`` column. Must be awaited inside a running event loop.

    Args:
        pool: The connection pool used to obtain a single connection for the query.

    Returns:
        The database revision as an ``int``, which is greater than or equal to ``REQUIRED_REVISION``.

    Raises:
        SchemaMismatch: If ``alembic_version`` holds zero rows or more than one row; if the single ``version_num``
            is not a ``str`` or contains any character that is not a decimal digit (including ``None``, an empty
            string, whitespace or a sign); or if the parsed revision is less than ``REQUIRED_REVISION``.
        Exception: Whatever ``db_errors`` raises or lets through for a failure while acquiring the connection,
            executing the query or fetching rows, such as a missing ``alembic_version`` table. The exact types are
            conditional on ``db_errors``.

    """
    with db_errors():
        async with pool.connection() as conn:
            cursor = await conn.execute("SELECT version_num FROM alembic_version")
            rows = await cursor.fetchall()
    if len(rows) != 1:
        raise SchemaMismatch(f"expected one row in alembic_version, found {len(rows)}")
    version = rows[0][0]
    if not isinstance(version, str) or not version.isdecimal():
        raise SchemaMismatch(f"database revision {version!r} is not a number")
    revision = int(version)
    if revision < REQUIRED_REVISION:
        raise SchemaMismatch(
            f"database is at revision {revision:04d}; this build needs {REQUIRED_REVISION:04d} or later"
        )
    return revision
