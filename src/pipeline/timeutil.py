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
    MIN_SPLIT: The shortest half ``TimeFrame.halve`` produces when splitting by time, one minute.
    TimeFrame: A non-empty, half-open span of UTC instants with the time zone that maps it to local dates.

Requirements:
    Python 3.14 or later: ``datetime.UTC`` needs 3.11, and the unquoted ``TimeFrame`` return annotations inside its own
    class body rely on the deferred evaluation of annotations added in 3.14.
    Time zone data available to ``zoneinfo``, from the host system or the ``tzdata`` package (required on Windows).
    Without it, importing this module raises ``zoneinfo.ZoneInfoNotFoundError`` when ``LONDON`` is created.

Example:
    >>> from datetime import date
    >>> start, end = day_window(date(2026, 10, 25), LONDON)
    >>> end - start
    datetime.timedelta(days=1, seconds=3600)

"""

from dataclasses import dataclass
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
    (``time.min``) in ``zone`` with ``fold=0``, converted to ``UTC``. When local midnight falls in a gap, it is read
    with the offset in force before the gap, which is the transition instant when the gap starts at midnight. ``end``
    is computed the same way as ``start`` for the following date, so windows for consecutive dates are contiguous and
    do not overlap. A window is 24 hours long except across an offset change,
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
        ``seconds`` is a count of seconds, not milliseconds, since 1970-01-01T00:00:00 UTC. A millisecond value is not
        detected unless it falls outside the supported range.

    Args:
        seconds: The Unix timestamp to convert, in seconds.

    Returns:
        An aware ``datetime`` with ``tzinfo`` set to ``UTC``, representing ``seconds`` seconds after the Unix epoch.
        ``0`` returns 1970-01-01 00:00:00+00:00.

    Raises:
        ValueError: If ``seconds`` is NaN or outside the supported range (years 1 to 9999 and the platform's
            ``time_t``), for example ``10**12``, ``10**20`` or infinity. ``OverflowError`` and ``OSError`` from the
            platform are re-raised as ``ValueError``.
        TypeError: If ``seconds`` is not a number.

    """
    try:
        return datetime.fromtimestamp(seconds, UTC)
    except (OverflowError, OSError) as err:
        raise ValueError(f"timestamp out of range: {seconds!r}") from err


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


#: The shortest half ``TimeFrame.halve`` produces when splitting by time rather than by whole days.
MIN_SPLIT: Final = timedelta(minutes=1)  # the shortest half an instant split may produce


@dataclass(frozen=True, slots=True)
class TimeFrame:
    """A non-empty, half-open span of UTC instants, with the time zone whose local days it is read in.

    The frame covers instants ``t`` with ``start <= t < end``. Instances are immutable; both bounds are normalised to
    ``UTC`` on construction, so frames built from the same instants in different offsets, with equal ``zone`` values,
    compare equal.

    Attributes:
        start: The first instant in the frame, aware, with ``tzinfo`` set to ``UTC``.
        end: The first instant after the frame, aware, with ``tzinfo`` set to ``UTC``. Always later than ``start``.
        zone: The time zone used to map the frame to local calendar dates, by ``dates`` and ``halve``. It does not
            affect ``start`` or ``end``.

    """

    start: datetime  # inclusive, UTC
    end: datetime  # exclusive, UTC
    zone: ZoneInfo  # the source's home zone

    def __post_init__(self) -> None:
        """Normalise both bounds to UTC and reject an empty frame.

        Runs after the dataclass ``__init__``. ``start`` and ``end`` are replaced with their ``ensure_utc`` values
        through ``object.__setattr__``, because the dataclass is frozen. ``zone`` is stored as given and not checked.

        Preconditions:
            None.

        Returns:
            None.

        Raises:
            ValueError: If ``start`` is not earlier than ``end`` after conversion to UTC, or if either bound is naive.
            TypeError: If either bound is not a ``datetime``.
            OverflowError: If converting a bound to UTC falls outside the ``datetime`` range.
            All but the empty-frame ``ValueError`` are raised by ``ensure_utc``.

        """
        object.__setattr__(self, "start", ensure_utc(self.start))
        object.__setattr__(self, "end", ensure_utc(self.end))
        if self.start >= self.end:
            raise ValueError(f"empty time frame: {self.start.isoformat()} >= {self.end.isoformat()}")

    @classmethod
    def days(cls, first: date, end: date, zone: ZoneInfo) -> TimeFrame:
        """Build the frame covering whole local calendar days from ``first`` up to, but not including, ``end``.

        Both bounds are local midnight in ``zone``, taken from ``day_window``. ``TimeFrame.days(d, d + 1 day, zone)``
        covers the single day ``d``.

        Preconditions:
            ``end`` is earlier than ``date.max``, because ``day_window`` also computes the following midnight.

        Args:
            first: The first local date in the frame.
            end: The first local date after the frame. It must be later than ``first``.
            zone: The time zone whose local midnights bound the frame. It is also stored as the frame's ``zone``.

        Returns:
            A ``TimeFrame`` from local midnight at the start of ``first`` to local midnight at the start of ``end``.

        Raises:
            ValueError: If the local midnights of ``first`` and ``end`` are the same instant or in the wrong order,
                raised by ``__post_init__`` as an empty frame. This happens when ``end`` is not later than ``first``,
                and also when ``zone`` skips every date from ``first`` up to ``end``, for example
                ``TimeFrame.days(date(2011, 12, 30), date(2011, 12, 31), ZoneInfo("Pacific/Apia"))``.
            TypeError: If ``zone`` is ``None``, or if ``first`` or ``end`` is a ``datetime``, raised by ``day_window``.
            OverflowError: If either date is ``date.max`` or a bound falls outside the ``datetime`` range, raised by
                ``day_window``.

        """
        return cls(day_window(first, zone)[0], day_window(end, zone)[0], zone)

    @property
    def dates(self) -> tuple[date, date] | None:
        """Return the frame as local dates, if it is made of whole days in ``zone``.

        The frame counts as whole days when both ``start`` and ``end`` are exactly local midnight in ``zone``, as
        ``day_window`` computes it. This inverts ``TimeFrame.days`` up to skipped dates: when ``zone`` skips a whole
        date, the midnights on either side of it are the same instant and the later date is returned, so
        ``TimeFrame.days(date(2011, 12, 29), date(2011, 12, 30), ZoneInfo("Pacific/Apia")).dates`` is
        ``(date(2011, 12, 29), date(2011, 12, 31))``, which covers the same instants.

        Preconditions:
            None.

        Returns:
            A tuple ``(first, end)`` where ``first`` is the first local date in the frame and ``end`` is the first
            local date after it, so ``TimeFrame.days(first, end, zone)`` equals this frame. ``None`` if either bound
            is not a local midnight.

        Raises:
            TypeError: If ``zone`` is ``None`` or not a ``tzinfo``, raised by ``business_date``.
            OverflowError: If a bound converted to ``zone``, or the midnight after its local date, falls outside the
                ``datetime`` range, raised by ``business_date`` or ``day_window``.

        """
        first = business_date(self.start, self.zone)
        end = business_date(self.end, self.zone)
        if day_window(first, self.zone)[0] != self.start or day_window(end, self.zone)[0] != self.end:
            return None
        return first, end

    def halve(self, *, whole_days: bool) -> tuple[TimeFrame, TimeFrame] | None:
        """Split the frame into two adjacent frames that together cover it exactly.

        With ``whole_days``, the split falls on a local midnight and both halves are whole days; with an odd number of
        days the first half gets the smaller share. Otherwise the split falls at ``start`` plus half the duration,
        floored to whole seconds, so the second half is never shorter than the first. Both halves keep ``zone``.

        Preconditions:
            None.

        Args:
            whole_days: Whether the halves must be whole local days in ``zone``. Pass ``True`` for connectors that only
                accept dates.

        Returns:
            A tuple ``(left, right)`` with ``left.start == self.start``, ``left.end == right.start`` and
            ``right.end == self.end``. ``None`` if the frame is too small to split: fewer than two days with
            ``whole_days``, or a first half shorter than ``MIN_SPLIT`` (a frame under two minutes) without it. Also
            ``None`` with ``whole_days`` when the middle date, ``first + days // 2`` counted in calendar dates, is one
            that ``zone`` skips, for example ``TimeFrame.days(date(2011, 12, 29), date(2011, 12, 31),
            ZoneInfo("Pacific/Apia")).halve(whole_days=True)``. This holds even when both halves would be non-empty.

        Raises:
            ValueError: If ``whole_days`` is true and the frame does not start and end at local midnight in ``zone``.
            TypeError: If ``whole_days`` is true and ``zone`` is ``None`` or not a ``tzinfo``, raised by ``dates``.
            OverflowError: If ``whole_days`` is true and a bound falls outside the ``datetime`` range, raised by
                ``dates``.

        """
        if whole_days:
            dates = self.dates
            if dates is None:
                raise ValueError("a whole-day split needs a frame that starts and ends at local midnight")
            first, end = dates
            days = (end - first).days
            if days < 2:
                return None
            middle = first + timedelta(days=days // 2)
            middle_start, middle_end = day_window(middle, self.zone)
            if middle_start == middle_end:  # zone skips the middle date, so a half could be empty
                return None
            return TimeFrame.days(first, middle, self.zone), TimeFrame.days(middle, end, self.zone)
        half = timedelta(seconds=(self.end - self.start) // timedelta(seconds=2))
        if half < MIN_SPLIT:
            return None
        middle = self.start + half
        return TimeFrame(self.start, middle, self.zone), TimeFrame(middle, self.end, self.zone)
