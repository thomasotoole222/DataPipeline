"""Put a time frame onto a connector request as the parameters an endpoint filters by.

Public API:
    Window: The protocol every window style implements: whether it needs whole days, and how it adds a frame.
    UnixSecondsWindow: A window sent as two top-level parameters holding whole Unix seconds.
    DateRangeWindow: A window sent as a ``{from, to}`` object of ISO dates at a key path in the body.
    FlatDateWindow: A window sent as two top-level parameters holding ISO dates.

"""

from dataclasses import dataclass
from datetime import timedelta
from typing import ClassVar, Protocol

from pipeline.connectors.spec import ConnectorRequest, KeyPath, Place, check_key_path
from pipeline.timeutil import TimeFrame, to_epoch


class Window(Protocol):
    """Describe how an endpoint receives a time frame, and whether that frame must be whole local days.

    ``whole_days`` tells the caller how to split a frame (``TimeFrame.halve(whole_days=...)``), so that every frame
    passed to ``apply`` is one the window can express. A window whose ``whole_days`` is ``True`` must reject, with
    ``ValueError`` from ``apply``, a frame that is not whole local days, rather than send a widened or truncated range.

    """

    @property
    def whole_days(self) -> bool:
        """Return whether the window can only express frames of whole local days.

        Preconditions:
            None.

        Returns:
            ``True`` when ``apply`` needs a frame that starts and ends at local midnight in the frame's zone.

        Raises:
            Nothing.

        """
        ...  # True: the frame must be whole local days, and splits keep it so

    def apply(self, request: ConnectorRequest, frame: TimeFrame) -> ConnectorRequest:
        """Return a copy of a request with a time frame added as filter parameters.

        Preconditions:
            None; a frame the window cannot express is rejected.

        Args:
            request: The request without window parameters.
            frame: The span of time to ask for.

        Returns:
            A copy of ``request`` with the window parameters added.

        Raises:
            ValueError: When ``frame`` cannot be expressed by the window, which for a ``whole_days`` window includes
                any frame that does not start and end at local midnight in its zone, or ``request`` cannot take its
                parameters; each implementation states the cases.
            TypeError: As an implementation states, for a frame whose zone is not a ``tzinfo``.
            OverflowError: As an implementation states, for a bound near the ``datetime`` range.

        """
        ...


@dataclass(frozen=True, slots=True)
class UnixSecondsWindow:
    """Send a frame as a start and an end parameter, each in whole Unix seconds.

    Both bounds are floored to the second by ``to_epoch``, so a frame with fractional bounds is sent as the
    whole-second range ``[floor(start), floor(end))``; adjacent frames share a bound and so still meet exactly. The end
    is the frame's exclusive end, so the endpoint must read the end parameter as exclusive. Construction guarantees
    that both parameter names are non-empty and different.

    Attributes:
        whole_days (ClassVar[bool]): Always ``False``; any frame whose bounds floor to different seconds can be sent,
            which includes every frame of at least one second.
        start_param (str): The name of the start parameter.
        end_param (str): The name of the end parameter.
        place (Place): Whether both parameters go in the query or at the top level of the body. Defaults to
            ``Place.BODY``.

    """

    whole_days: ClassVar[bool] = False

    start_param: str  # e.g. "update_time_ge"
    end_param: str  # e.g. "update_time_lt"; exclusive, as the frame
    place: Place = Place.BODY

    def __post_init__(self) -> None:
        """Reject an empty or shared parameter name.

        Preconditions:
            None.

        Returns:
            None.

        Raises:
            ValueError: When ``start_param`` or ``end_param`` is empty or they are equal.

        """
        # a bad name would otherwise fail only when a task runs
        if not self.start_param or not self.end_param or self.start_param == self.end_param:
            raise ValueError(f"parameter names must be non-empty and differ: {self.start_param!r}, {self.end_param!r}")

    def apply(self, request: ConnectorRequest, frame: TimeFrame) -> ConnectorRequest:
        """Return a copy of a request with the frame's bounds added as Unix seconds.

        The values are ``int``: a JSON number in the body, a decimal string in the query.

        Preconditions:
            None.

        Args:
            request: The request without window parameters.
            frame: The span to send. ``TimeFrame`` guarantees ``start < end``.

        Returns:
            A new ``ConnectorRequest`` with ``start_param`` set to ``to_epoch(frame.start)`` and ``end_param`` to
            ``to_epoch(frame.end)``.

        Raises:
            ValueError: When both bounds floor to the same second, so the range of whole seconds would be empty; or
                from ``ConnectorRequest.with_param`` when a name is already set on ``request``, or ``place`` is
                ``Place.BODY`` and ``request`` is a GET. The names themselves always pass ``check_key_path``, as
                ``__post_init__`` guarantees. ``with_param``'s ``copy.deepcopy`` failure cannot occur: the values are
                ``int`` and the rest of the body was already deep-copied when ``request`` was built.

        """
        start, end = to_epoch(frame.start), to_epoch(frame.end)
        if start >= end:
            raise ValueError(f"time frame floors to an empty range of whole seconds: {frame}")
        return request.with_param(self.place, (self.start_param,), start).with_param(self.place, (self.end_param,), end)


@dataclass(frozen=True, slots=True)
class DateRangeWindow:
    """Send a frame of whole local days as an object of two ISO dates at a key path in the body.

    The object is ``{from_key: first, to_key: end}``, and ``end`` is the last day of the frame when
    ``end_inclusive`` is true, or the day after it when false. Construction guarantees that ``at`` is a valid key
    path. Construction also guarantees that ``from_key`` and ``to_key`` are non-empty and different.

    Attributes:
        whole_days (ClassVar[bool]): Always ``True``; the frame must start and end at local midnight.
        at (KeyPath): Where the date object goes in the body. Checked by ``check_key_path``.
        end_inclusive (bool): Whether the endpoint reads the end date as the last day included.
        from_key (str): The key of the first date in the object. Defaults to ``"from"``.
        to_key (str): The key of the end date in the object. Defaults to ``"to"``.

    """

    whole_days: ClassVar[bool] = True

    at: KeyPath  # where the {from, to} object goes in the body, e.g. ("search_params", "date_range")
    end_inclusive: bool  # True: "to" is the last day included; False: the first day excluded
    from_key: str = "from"
    to_key: str = "to"

    def __post_init__(self) -> None:
        """Check that ``at`` is a valid key path and the date keys are non-empty and different.

        Preconditions:
            None.

        Returns:
            None.

        Raises:
            ValueError: From ``check_key_path`` when ``at`` is empty or contains an empty key; or when ``from_key``
                or ``to_key`` is empty or they are equal (equal keys would build a one-key object that drops the
                first date).

        """
        check_key_path(self.at)
        # equal keys would build a one-key object and silently drop the first date
        if not self.from_key or not self.to_key or self.from_key == self.to_key:
            raise ValueError(f"date keys must be non-empty and differ: {self.from_key!r}, {self.to_key!r}")

    def apply(self, request: ConnectorRequest, frame: TimeFrame) -> ConnectorRequest:
        """Return a copy of a request with the frame's dates set as an object at ``at`` in the body.

        Preconditions:
            None.

        Args:
            request: The request without window parameters. It must be a POST for the call to succeed.
            frame: The span to send, which must be whole local days in its zone.

        Returns:
            A new ``ConnectorRequest`` with ``{from_key: first, to_key: end}`` at ``at``, the dates as ISO strings
            from ``_date_bounds``.

        Raises:
            ValueError: From ``_date_bounds`` when ``frame`` is not whole local days; or from
                ``ConnectorRequest.with_body_at`` when ``request`` is a GET, ``at`` already holds a value, or an
                intermediate key holds something that is not a mapping.
            TypeError: From ``TimeFrame.dates``, through ``_date_bounds``, when the frame's zone is not a ``tzinfo``.
            OverflowError: From ``TimeFrame.dates``, through ``_date_bounds``, for a bound near the ``datetime`` range.

        """
        first, end = _date_bounds(frame, end_inclusive=self.end_inclusive)
        return request.with_body_at(self.at, {self.from_key: first, self.to_key: end})


@dataclass(frozen=True, slots=True)
class FlatDateWindow:
    """Send a frame of whole local days as two parameters holding ISO dates.

    The end parameter is the last day of the frame when ``end_inclusive`` is true, or the day after it when false.
    Construction guarantees that both parameter names are non-empty and different.

    Attributes:
        whole_days (ClassVar[bool]): Always ``True``; the frame must start and end at local midnight.
        from_param (str): The name of the first-date parameter.
        to_param (str): The name of the end-date parameter.
        end_inclusive (bool): Whether the endpoint reads the end date as the last day included.
        place (Place): Whether both parameters go in the query or at the top level of the body. Defaults to
            ``Place.BODY``.

    """

    whole_days: ClassVar[bool] = True

    from_param: str  # e.g. "time_from"
    to_param: str  # e.g. "time_to"
    end_inclusive: bool
    place: Place = Place.BODY

    def __post_init__(self) -> None:
        """Reject an empty or shared parameter name.

        Preconditions:
            None.

        Returns:
            None.

        Raises:
            ValueError: When ``from_param`` or ``to_param`` is empty or they are equal.

        """
        if not self.from_param or not self.to_param or self.from_param == self.to_param:
            raise ValueError(f"parameter names must be non-empty and differ: {self.from_param!r}, {self.to_param!r}")

    def apply(self, request: ConnectorRequest, frame: TimeFrame) -> ConnectorRequest:
        """Return a copy of a request with the frame's dates added as two parameters.

        Preconditions:
            None.

        Args:
            request: The request without window parameters.
            frame: The span to send, which must be whole local days in its zone.

        Returns:
            A new ``ConnectorRequest`` with ``from_param`` and ``to_param`` set to the ISO dates from ``_date_bounds``.

        Raises:
            ValueError: From ``_date_bounds`` when ``frame`` is not whole local days; or from
                ``ConnectorRequest.with_param`` when a name is already set on ``request``, or ``place`` is
                ``Place.BODY`` and ``request`` is a GET. The names always pass ``check_key_path``, as ``__post_init__``
                guarantees, and the ``copy.deepcopy`` failure ``with_param`` allows cannot occur: the values are
                strings and the rest of the body was already deep-copied when ``request`` was built.
            TypeError: From ``TimeFrame.dates``, through ``_date_bounds``, when the frame's zone is not a ``tzinfo``.
            OverflowError: From ``TimeFrame.dates``, through ``_date_bounds``, for a bound near the ``datetime`` range.

        """
        first, end = _date_bounds(frame, end_inclusive=self.end_inclusive)
        return request.with_param(self.place, (self.from_param,), first).with_param(self.place, (self.to_param,), end)


def _date_bounds(frame: TimeFrame, *, end_inclusive: bool) -> tuple[str, str]:
    """Return a whole-day frame's first date and end date as ISO strings.

    ``TimeFrame`` is non-empty and ``dates`` returns only local midnights, so the end date is at least a day after
    the first; with ``end_inclusive`` the end moves back one day and is never before the first, and cannot
    underflow ``date.min``.

    Preconditions:
        None.

    Args:
        frame: The frame to convert.
        end_inclusive: Whether to return the last day in the frame rather than the day after it.

    Returns:
        ``(first, end)`` as ``YYYY-MM-DD`` strings: the first local date, and either the last local date
        (``end_inclusive``) or the first local date after the frame.

    Raises:
        ValueError: When ``frame`` does not start and end at local midnight in its zone.
        TypeError: From ``TimeFrame.dates`` when the frame's zone is not a ``tzinfo``.
        OverflowError: From ``TimeFrame.dates`` when a bound or the following midnight is outside the ``datetime``
            range.

    """
    dates = frame.dates
    if dates is None:
        raise ValueError(f"a date window needs a frame of whole local days: {frame}")
    first, end = dates
    if end_inclusive:
        end -= timedelta(days=1)  # the half-open frame's last day
    return first.isoformat(), end.isoformat()
