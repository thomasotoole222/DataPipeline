"""The warehouse connection pool and the schema revision check run at startup.

Public API:
    Connection: An async psycopg connection that returns tuple rows.
    Pool: An async connection pool of ``Connection`` objects.
    REQUIRED_REVISION: The lowest database schema revision this build accepts.
    DEAD_CONNECTION_PARAMS: libpq parameters that detect a dead connection.
    open_pool: Open the warehouse connection pool and wait until it is ready.
    check_schema: Check that the database schema revision meets ``REQUIRED_REVISION``.
    storable_text: Return text with every character Postgres text and jsonb cannot store replaced with U+FFFD.

"""

from collections.abc import Mapping
from datetime import timedelta
from typing import Final

from psycopg import AsyncConnection, sql
from psycopg.rows import TupleRow
from psycopg_pool import AsyncConnectionPool

from pipeline.config import Settings
from pipeline.errors import SchemaMismatch, db_errors

type Connection = AsyncConnection[TupleRow]
type Pool = AsyncConnectionPool[Connection]

#: Lowest ``alembic_version`` revision of the database schema that this build accepts.
REQUIRED_REVISION: Final = 0
#: Longest ``version_num`` accepted: Alembic revision ids are at most 32 characters (its ``VARCHAR(32)`` column).
_MAX_REVISION_LENGTH: Final = 32

#: Longest wait for the pool to open, and for a connection to be checked out of it.
_POOL_TIMEOUT: Final = timedelta(seconds=10)
#: Server-side ``statement_timeout`` set on every pool connection.
_STATEMENT_TIMEOUT: Final = timedelta(seconds=60)
#: Server-side ``idle_in_transaction_session_timeout`` set on every pool connection.
_IDLE_IN_TRANSACTION_TIMEOUT: Final = timedelta(seconds=60)
#: Session ``TimeZone`` set by ``_configure`` on every pool connection, so a ``timestamptz`` literal without an offset
#: reads the same on any server and whatever the client environment (``PGTZ``) asks for.
_TIME_ZONE: Final = "UTC"
#: Idle time on a connection before the first TCP keepalive probe is sent.
_KEEPALIVES_IDLE: Final = timedelta(seconds=60)
#: Time between unanswered TCP keepalive probes.
_KEEPALIVES_INTERVAL: Final = timedelta(seconds=15)
#: Unanswered TCP keepalive probes after which the connection is treated as dead.
_KEEPALIVES_COUNT: Final = 4
#: Longest time sent data may stay unacknowledged before the connection is treated as dead.
_TCP_USER_TIMEOUT: Final = timedelta(seconds=120)


def storable_text(text: str) -> str:
    """Return ``text`` with every character Postgres ``text`` and ``jsonb`` cannot store replaced with U+FFFD.

    Postgres stores only Unicode scalar values other than NUL: U+0001 to U+D7FF and U+E000 to U+10FFFF. Every other
    character of a Python ``str`` is NUL (U+0000) or a surrogate (U+D800 to U+DFFF), which psycopg cannot encode and
    ``jsonb`` refuses; each one is replaced with U+FFFD, one for one, so the length is unchanged. A surrogate pair held
    as two characters is two surrogates, and both are replaced.

    Preconditions:
        None.

    Args:
        text: The text to clean.

    Returns:
        ``text`` itself when every character is storable, otherwise a new string with each character outside the
        storable set replaced with U+FFFD.

    Raises:
        Nothing.

    """
    if "\x00" not in text and (text.isascii() or not any("\ud800" <= c <= "\udfff" for c in text)):
        return text
    return "".join("\ufffd" if c == "\x00" or "\ud800" <= c <= "\udfff" else c for c in text)


def _ms(value: timedelta) -> int:
    """Convert a duration to whole milliseconds.

    Divides ``value`` by one millisecond with ``timedelta`` floor division, which is exact integer arithmetic on
    microseconds, so there is no floating-point rounding error at any size.

    Preconditions:
        None.

    Args:
        value: The duration to convert.

    Returns:
        ``value // timedelta(milliseconds=1)``: the whole number of milliseconds in ``value``, rounded towards
        negative infinity. A positive duration loses any sub-millisecond remainder; a negative duration with a
        remainder rounds down (``timedelta(microseconds=-1500)`` gives ``-2``). Every caller on this page passes a
        positive whole number of seconds, so the result is exact there.

    Raises:
        Nothing.

    """
    return value // timedelta(milliseconds=1)


#: libpq parameters that detect a dead connection in about 2 minutes, used by the pool and by
#: ``db.control.open_listener``. An idle connection is dead after 60 s idle plus 4 unanswered probes 15 s apart
#: (120 s); a connection with unacknowledged sent data is dead after ``tcp_user_timeout`` of 120 000 ms. The keepalive
#: values are in whole seconds and ``tcp_user_timeout`` in milliseconds, the units libpq expects.
DEAD_CONNECTION_PARAMS: Final[Mapping[str, int]] = {
    "keepalives": 1,
    "keepalives_idle": _KEEPALIVES_IDLE // timedelta(seconds=1),
    "keepalives_interval": _KEEPALIVES_INTERVAL // timedelta(seconds=1),
    "keepalives_count": _KEEPALIVES_COUNT,
    "tcp_user_timeout": _ms(_TCP_USER_TIMEOUT),
}


async def _configure(conn: Connection) -> None:
    """Set a new pool connection's session ``TimeZone`` to ``_TIME_ZONE``.

    Runs ``SET TIME ZONE`` after the connection is made, so it overrides a ``TimeZone`` from the server, database or
    role defaults and from the client's ``PGTZ``, which libpq sends at connect time. The pool's connections use
    autocommit, so the setting takes effect at once and no transaction is left open.

    Preconditions:
        ``conn`` is a new, open, autocommit connection, as ``open_pool``'s ``kwargs`` make every pool connection.

    Args:
        conn: The connection the pool has just made.

    Returns:
        None.

    Raises:
        psycopg.Error: Any error from ``SET TIME ZONE``, unchanged; the pool then discards the connection and tries
            again, so ``open_pool`` and a checkout see it as ``PoolTimeout``.

    """
    await conn.execute(sql.SQL("SET TIME ZONE {}").format(sql.Literal(_TIME_ZONE)))


async def open_pool(settings: Settings) -> Pool:
    """Open the warehouse connection pool and wait until it is ready.

    Every connection the pool creates uses autocommit, takes ``settings.worker_id`` as its ``application_name``, and
    sets ``statement_timeout`` and ``idle_in_transaction_session_timeout`` through the ``options`` connection
    parameter. Every new connection is then configured by ``_configure``, which sets its session ``TimeZone`` to
    ``_TIME_ZONE`` (UTC) after connecting, so neither a server, database or role default nor a client ``PGTZ``
    changes it. Every connection also takes the libpq parameters in ``DEAD_CONNECTION_PARAMS`` (TCP keepalives and
    ``tcp_user_timeout``), so a connection whose server has gone is found dead in about 2 minutes. Every checkout runs
    ``AsyncConnectionPool.check_connection`` first.

    psycopg merges these keyword parameters over the URL's own (``make_conninfo``), so a value the URL set for
    ``options``, ``application_name``, ``keepalives*`` or ``tcp_user_timeout`` would be replaced, not combined;
    ``Settings`` therefore refuses a ``warehouse_url`` that sets any of them. ``PGOPTIONS`` is never read.

    If ``pool.open`` fails or is interrupted, the pool is closed before the exception propagates. A connection that
    cannot be made (a wrong password, say) is retried by the pool until ``_POOL_TIMEOUT``, so it surfaces as a
    ``PoolTimeout`` mapped to ``DatabaseUnavailable``, not as the connection error itself.

    Preconditions:
        Must be awaited inside a running event loop. ``settings.concurrency`` is 1 to 4 and ``settings.worker_id``
        is non-empty, both guaranteed by ``Settings`` validation, so ``max_size`` (2 to 8) is never below
        ``min_size``. ``settings.warehouse_url`` starts with ``postgresql://`` or ``postgres://`` and sets none of
        the parameters named above (``Settings``).

    Args:
        settings: Supplies the connection string (``warehouse_url``), the pool's maximum size
            (``2 * concurrency``) and each connection's ``application_name`` (``worker_id``).

    Returns:
        The opened pool, named ``"warehouse"``, with ``min_size`` 1, ``max_size`` ``2 * settings.concurrency`` and a
        checkout timeout of ``_POOL_TIMEOUT``.

    Raises:
        Exception: Any error from the ``AsyncConnectionPool`` constructor, unchanged, since it runs outside
            ``db_errors``.
        DatabaseUnavailable: When ``pool.open`` does not have ``min_size`` connections within ``_POOL_TIMEOUT``
            (``PoolTimeout``), or raises another error in ``db_errors``' unavailable set or a bare
            ``psycopg.OperationalError``, after ``pool.close``.
        BaseException: Any other exception from ``pool.open``, ``asyncio.CancelledError`` included, after ``db_errors``
            and then ``pool.close`` have run. If ``pool.close`` itself raises, its exception propagates instead, with
            the original as its ``__context__``.

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
        configure=_configure,
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
    ``version_num`` is a ``str`` of at most ``_MAX_REVISION_LENGTH`` characters made only of the ASCII digits ``0``
    to ``9``. That string is converted with ``int`` and compared with ``REQUIRED_REVISION``.

    Preconditions:
        ``pool`` must be open and able to hand out a connection. The connected database must contain an
        ``alembic_version`` table with a ``version_num`` column. Must be awaited inside a running event loop.

    Args:
        pool: The connection pool used to obtain a single connection for the query.

    Returns:
        The database revision as an ``int``, which is greater than or equal to ``REQUIRED_REVISION``.

    Raises:
        SchemaMismatch: If ``alembic_version`` holds zero rows or more than one row; if the single ``version_num``
            is not a ``str`` or contains any character that is not an ASCII digit (including ``None``, an empty
            string, whitespace, a sign or a decimal digit from outside ASCII, such as an Arabic-Indic or fullwidth
            digit); if it is longer than ``_MAX_REVISION_LENGTH`` characters (its text is then not in the message);
            or if the parsed revision is less than ``REQUIRED_REVISION``.
        DatabaseUnavailable: When no connection can be checked out within ``_POOL_TIMEOUT`` (``PoolTimeout``), or the
            checkout, query or fetch raises another error in ``db_errors``' unavailable set or a bare
            ``psycopg.OperationalError``.
        DatabaseConflict: When the query raises ``SerializationFailure`` or ``DeadlockDetected`` (``db_errors``).
        Exception: Any other error from acquiring the connection, executing the query or fetching rows, unchanged by
            ``db_errors``, such as ``psycopg.errors.UndefinedTable`` for a missing ``alembic_version`` table or
            ``psycopg.errors.QueryCanceled`` when the query outlasts ``_STATEMENT_TIMEOUT``.
        BaseException: ``asyncio.CancelledError`` or any other non-``Exception`` raised while the connection is
            acquired or used, unchanged; a connection already checked out is returned to the pool.

    """
    with db_errors():
        async with pool.connection() as conn:
            cursor = await conn.execute("SELECT version_num FROM alembic_version")
            rows = await cursor.fetchall()
    if len(rows) != 1:
        raise SchemaMismatch(f"expected one row in alembic_version, found {len(rows)}")
    version = rows[0][0]
    if isinstance(version, str) and len(version) > _MAX_REVISION_LENGTH:
        raise SchemaMismatch(f"database revision is {len(version)} characters; at most {_MAX_REVISION_LENGTH} allowed")
    if not isinstance(version, str) or not (version.isascii() and version.isdigit()):
        raise SchemaMismatch(f"database revision {version!r} is not a number")
    revision = int(version)
    if revision < REQUIRED_REVISION:
        raise SchemaMismatch(
            f"database is at revision {revision:04d}; this build needs {REQUIRED_REVISION:04d} or later"
        )
    return revision
