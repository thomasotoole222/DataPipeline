"""Time zone aware helpers for UTC instants, local calendar days and Unix timestamps.

Every ``datetime`` returned by this module is aware, with ``tzinfo`` set to ``UTC``. Naive datetimes are rejected
rather than guessed at, so callers must attach a time zone before passing a value in.

Public API:
    LONDON: The ``Europe/London`` time zone as a ``ZoneInfo``.
    utc_now: Return the current instant in UTC.
    ensure_utc: Convert an aware ``datetime`` to UTC and reject naive values.
    day_window: Return the half-open UTC interval ``[start, end)`` covering a calendar day in a given zone.
    business_date: Return the calendar date of an instant as observed in a given zone.
    from_epoch: Convert a Unix timestamp in seconds to an aware UTC ``datetime``.
    to_epoch: Convert an aware ``datetime`` to whole seconds since the Unix epoch, floored.

Requirements:
    Python 3.11 or later, for ``datetime.UTC``.
    Time zone data available to ``zoneinfo``, from the host system or the ``tzdata`` package (required on Windows).
    Without it, importing this module raises ``zoneinfo.ZoneInfoNotFoundError`` when ``LONDON`` is created.

Example:
    >>> from datetime import date
    >>> start, end = day_window(date(2026, 10, 25), LONDON)
    >>> end - start
    datetime.timedelta(days=1, seconds=3600)

"""

from datetime import UTC, date, datetime, time, timedelta
from typing import Final
from zoneinfo import ZoneInfo

#: The business time zone, ``Europe/London``.
LONDON: Final = ZoneInfo("Europe/London")
#: The Unix epoch, 1970-01-01T00:00:00 UTC, as an aware ``datetime``.
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def utc_now() -> datetime:
    """Return the current time as a timezone-aware ``datetime`` in UTC.

    Reads the system wall clock through ``datetime.now`` with ``UTC`` as the time zone, so the result carries ``tzinfo``
    set to ``UTC``. Successive calls are not guaranteed to be non-decreasing, because the wall clock can be adjusted.

    Preconditions:
        None.

    Args:
        None.

    Returns:
        A timezone-aware ``datetime`` for the current instant, with ``tzinfo`` equal to ``UTC``.

    Raises:
        None.

    """
    return datetime.now(UTC)


def ensure_utc(dt: datetime) -> datetime:
    """Convert an aware datetime to UTC.

    A naive ``dt`` is rejected, not converted. The result represents the same instant as ``dt``.

    Preconditions:
        None.

    Args:
        dt: The datetime to convert. It must be aware, meaning ``dt.utcoffset()`` returns a ``timedelta``.

    Returns:
        An aware ``datetime`` for the same instant as ``dt``, with ``tzinfo`` set to ``UTC``. If ``dt.tzinfo`` is
        already the ``UTC`` object itself, ``dt`` is returned as is.

    Raises:
        TypeError: If ``dt`` is not a ``datetime`` instance, for example a ``date`` or ``None``.
        ValueError: If ``dt`` is naive, that is ``dt.utcoffset()`` returns ``None``.
        OverflowError: If the converted value falls outside the ``datetime`` range, for example ``datetime.max``
            with an offset of ``-01:00``.
        Any exception raised by ``dt.tzinfo.utcoffset()`` propagates unchanged. No exception is swallowed.

    """
    if not isinstance(dt, datetime):
        raise TypeError(f"expected a datetime, not {type(dt).__name__}")
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("naive datetime")
    return dt.astimezone(UTC)


def day_window(d: date, zone: ZoneInfo) -> tuple[datetime, datetime]:
    """Return the UTC instants that bound a calendar day in a time zone.

    The window is half-open and covers instants ``t`` with ``start <= t < end``. Both bounds are local midnight
    (``time.min``) in ``zone`` with ``fold=0``, converted to ``UTC``. When local midnight falls in a gap, it resolves
    to the instant of the transition. ``end`` is computed the same way as ``start`` for the following date, so windows
    for consecutive dates are contiguous and do not overlap. A window is 24 hours long except across an offset change,
    where its length differs by the size of the change.

    Preconditions:
        ``d`` is earlier than ``date.max``.
        If ``d`` is ``date.min``, the offset of ``zone`` at local midnight on ``d`` is zero or negative.

    Args:
        d: The calendar date. A ``datetime`` is rejected, because its UTC date can differ from the local business
            date; convert it with ``business_date`` first.
        zone: The time zone whose local midnights bound the day.

    Returns:
        A tuple ``(start, end)`` of aware ``datetime`` objects with ``tzinfo`` set to ``UTC``: ``start`` is local
        midnight at the start of ``d`` and ``end`` is local midnight at the start of the following date. If ``zone``
        skips the whole of ``d``, ``start`` equals ``end``.

    Raises:
        TypeError: If ``zone`` is ``None``, or if ``d`` is a ``datetime`` rather than a ``date``.
        OverflowError: If ``d`` is ``date.max``, or if converting a bound to ``UTC`` falls outside the supported
            ``datetime`` range, for example ``date.min`` in a zone ahead of UTC. Raised by the standard library and
            passed through unchanged.

    """
    if zone is None:
        raise TypeError("zone must be a tzinfo, not None")
    if isinstance(d, datetime):
        raise TypeError("d must be a date, not a datetime; use business_date first")
    start = datetime.combine(d, time.min, tzinfo=zone)
    end = datetime.combine(d + timedelta(days=1), time.min, tzinfo=zone)
    return start.astimezone(UTC), end.astimezone(UTC)


def business_date(instant: datetime, zone: ZoneInfo) -> date:
    """Return the calendar date of ``instant`` as observed in ``zone``.

    Normalises ``instant`` with ``ensure_utc``, converts the result to ``zone`` and returns the date part of the
    converted value. The time of day and the time zone are discarded.

    Preconditions:
        ``instant`` is a value that ``ensure_utc`` accepts.
        The local time of ``instant`` in ``zone`` lies between ``datetime.min`` and ``datetime.max``.

    Args:
        instant: The moment to convert.
        zone: The time zone whose wall calendar decides the date.

    Returns:
        The ``date`` shown on the wall calendar in ``zone`` at the moment ``instant``. For example, 2026-10-08 23:30
        UTC in ``ZoneInfo("Europe/London")`` returns 2026-10-09.

    Raises:
        TypeError: If ``zone`` is ``None`` or is not a ``tzinfo`` instance.
        OverflowError: If converting to ``zone`` moves the value outside the range of ``datetime``.
        Any exception raised by ``ensure_utc`` propagates unchanged. No exceptions are swallowed.

    """
    if zone is None:
        raise TypeError("zone must be a tzinfo instance, not None")
    return ensure_utc(instant).astimezone(zone).date()


def from_epoch(seconds: int) -> datetime:
    """Convert a Unix timestamp in seconds to an aware UTC ``datetime``.

    Preconditions:
        ``seconds`` is a count of seconds, not milliseconds, since 1970-01-01T00:00:00 UTC.
        ``seconds`` maps to a date between years 1 and 9999 and fits the platform's ``time_t``.

    Args:
        seconds: The Unix timestamp to convert, in seconds.

    Returns:
        An aware ``datetime`` with ``tzinfo`` set to ``UTC``, representing ``seconds`` seconds after the Unix epoch.
        ``0`` returns 1970-01-01 00:00:00+00:00.

    Raises:
        ValueError: If ``seconds`` is NaN or outside the supported range, for example ``10**12``, ``10**20`` or
            infinity. ``OverflowError`` and ``OSError`` from the platform are re-raised as ``ValueError``.
        TypeError: If ``seconds`` is not a number.

    """
    try:
        return datetime.fromtimestamp(seconds, UTC)
    except (OverflowError, OSError) as exc:
        raise ValueError(f"timestamp out of range: {seconds!r}") from exc


def to_epoch(dt: datetime) -> int:
    """Convert a datetime to whole seconds since the Unix epoch.

    The offset from ``_EPOCH`` is computed with integer ``timedelta`` arithmetic and floored, so fractional seconds
    always round towards the earlier instant, on both sides of the epoch.

    Preconditions:
        None.

    Args:
        dt: The instant to convert. Handling of naive and non-UTC values is determined by ``ensure_utc``.

    Returns:
        The number of whole seconds from 1970-01-01T00:00:00 UTC to ``ensure_utc(dt)``, floored, as an ``int``.
        Instants before the epoch give negative values; 1969-12-31T23:59:59.5 UTC gives ``-1``.

    Raises:
        TypeError: If ``dt`` is not a ``datetime``.
        ValueError: If ``dt`` is naive.
        OverflowError: As for ``ensure_utc``, if the value converted to UTC falls outside the ``datetime`` range.
        All are raised by ``ensure_utc``.

    """
    return (ensure_utc(dt) - _EPOCH) // timedelta(seconds=1)
