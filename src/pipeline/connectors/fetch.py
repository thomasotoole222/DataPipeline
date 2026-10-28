"""Fetch every page of one endpoint over a time frame, splitting the frame when a row limit is exceeded.

Public API:
    CURSOR_VERSION: The version number written into, and required of, every fetch cursor.
    Page: One fetched response page, with the cursors to fetch it again and to fetch the next page.
    fetch_pages: Fetch an endpoint's pages in order for a source, optionally from a resume cursor.

"""

import json
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from types import MappingProxyType
from typing import Any, Final, TypeIs
from uuid import UUID

from pipeline.connectors.client import ConnectorClient
from pipeline.connectors.spec import EndpointSpec
from pipeline.connectors.windows import Window
from pipeline.errors import RowLimitReached
from pipeline.timeutil import TimeFrame, ensure_utc, utc_now

#: The version stored under ``"v"`` in every fetch cursor. ``_Position.decode`` rejects any cursor whose ``"v"``
#: compares unequal to it.
CURSOR_VERSION: Final = 1
#: An empty, read-only mapping used as the default of ``fetch_pages``'s ``path_params``, so the default cannot be
#: mutated and shared between calls.
_NO_PATH_PARAMS: Final[Mapping[str, str]] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class Page:
    """Hold one fetched response page and the cursors around it.

    Construction guarantees that ``page_no`` and ``record_count`` are not negative, and replaces ``fetched_at`` with
    its UTC value. ``payload`` is stored as given, not copied. ``fetch_pages`` numbers pages from 0 with no gaps
    across sub-frames; a request whose response only caused a split is not a page and takes no number.

    Attributes:
        page_no (int): The page's position in the fetch, 0 or more.
        cursor_in (str | None): The cursor that resumes the fetch at this page, or ``None`` for the first page of a
            fetch started without a cursor. After a split it is the cursor from before the split, so resuming from
            it repeats the split request first.
        cursor_next (str | None): The cursor that resumes the fetch at the next page, or ``None`` on the last page.
        fetched_at (datetime): When the response was received, aware, in UTC.
        payload (dict[str, Any]): The decoded response body.
        record_count (int): The number of records in ``payload``, 0 or more.

    """

    page_no: int  # 0, 1, 2, ... across the whole task, splits included
    cursor_in: str | None  # None: the task's first page
    cursor_next: str | None  # None: the task's last page
    fetched_at: datetime
    payload: dict[str, Any]  # the body exactly as the client returned it
    record_count: int

    def __post_init__(self) -> None:
        """Reject a negative page number or record count, and normalise ``fetched_at`` to UTC.

        Preconditions:
            None.

        Returns:
            None.

        Raises:
            ValueError: When ``page_no`` or ``record_count`` is negative; or from ``ensure_utc`` when
                ``fetched_at`` is naive.
            OverflowError: From ``ensure_utc`` when converting ``fetched_at`` to UTC leaves the ``datetime`` range.

        """
        if self.page_no < 0:
            raise ValueError(f"page_no must be 0 or more: {self.page_no}")
        if self.record_count < 0:
            raise ValueError(f"record_count must be 0 or more: {self.record_count}")
        object.__setattr__(self, "fetched_at", ensure_utc(self.fetched_at))


@dataclass(frozen=True, slots=True)
class _Position:
    """Hold where a fetch is: the next page number, the paging cursor and the sub-frames still to fetch.

    ``bounds`` lists the edges of the remaining sub-frames in order, so the current sub-frame is
    ``[bounds[0], bounds[1])`` and each later pair of neighbours is one more sub-frame. It is empty for a fetch
    with no time frame. Nothing is checked on construction; ``start``, ``decode``, ``advance``, ``next_frame`` and
    ``split`` only build positions whose ``bounds`` is empty without a frame, or has at least two strictly
    increasing edges ending at the frame's end with one.

    Attributes:
        page_no (int): The number the next page will get.
        paging (str | None): The paginator cursor for the next request, or ``None`` for a sub-frame's first page.
        rows (int): Records fetched so far in the current sub-frame, before the next request.
        bounds (tuple[datetime, ...]): The edges of the remaining sub-frames, in UTC, or empty with no frame.

    """

    page_no: int  # of the next page to yield
    paging: str | None  # the paginator's own cursor; None: the sub-frame's first page, not yet checked for size
    rows: int  # records already yielded in the current sub-frame
    bounds: tuple[datetime, ...]  # sub-frames left, as boundaries: current = [0, 1), then [1, 2), ...; () unwindowed

    def sub_frame(self, frame: TimeFrame | None) -> TimeFrame | None:
        """Return the current sub-frame, in the zone of the whole frame.

        Preconditions:
            When ``frame`` is not ``None``, ``bounds`` has at least two edges; every constructor on this page
            ensures this.

        Args:
            frame: The fetch's whole frame, or ``None`` when it has none.

        Returns:
            ``TimeFrame(bounds[0], bounds[1], frame.zone)``, or ``None`` when ``frame`` is ``None``.

        Raises:
            IndexError: When ``frame`` is set and ``bounds`` has fewer than two edges, which breaks the precondition.

        """
        if frame is None:
            return None
        return TimeFrame(self.bounds[0], self.bounds[1], frame.zone)

    def advance(self, paging: str, rows: int) -> _Position:
        """Return the position of the next page in the same sub-frame.

        Preconditions:
            None.

        Args:
            paging: The paginator cursor of the next page.
            rows: Records fetched in the current sub-frame including the page just fetched.

        Returns:
            A position with the next page number, ``paging``, ``rows`` and the same ``bounds``.

        Raises:
            Nothing.

        """
        return _Position(self.page_no + 1, paging, rows, self.bounds)

    def next_frame(self) -> _Position | None:
        """Return the position of the first page of the next sub-frame, if there is one.

        Preconditions:
            None.

        Returns:
            A position with the next page number, no paging cursor, no rows and ``bounds`` without its first edge;
            or ``None`` when ``bounds`` has two edges or fewer, so the current sub-frame is the last (or there is no
            frame).

        Raises:
            Nothing.

        """
        if len(self.bounds) <= 2:  # the last sub-frame (or no window) is done
            return None
        return _Position(self.page_no + 1, None, 0, self.bounds[1:])

    def split(self, window: Window | None, frame: TimeFrame | None) -> _Position | None:
        """Return this position with the current sub-frame replaced by its two halves.

        The page number stays the same and the paging cursor and row count are reset, so the next request is the
        first page of the left half. The right half's start becomes a new edge in ``bounds``.

        Preconditions:
            When ``frame`` is not ``None``, ``bounds`` has at least two edges, as ``sub_frame`` needs. When
            ``window.whole_days`` is ``True``, the current sub-frame is whole local days; ``fetch_pages`` ensures
            this because ``window.apply`` has already accepted the same sub-frame.

        Args:
            window: The fetch's window, whose ``whole_days`` decides how the sub-frame is halved, or ``None``.
            frame: The fetch's whole frame, or ``None``.

        Returns:
            The split position, or ``None`` when ``window`` or ``frame`` is ``None`` or ``TimeFrame.halve`` finds the
            sub-frame too small to split.

        Raises:
            ValueError: From ``TimeFrame.halve`` when ``window.whole_days`` is ``True`` and the sub-frame is not whole
                local days, which breaks the precondition.
            TypeError: From ``TimeFrame.halve`` when ``window.whole_days`` is ``True`` and the zone is not a
                ``tzinfo``.
            OverflowError: From ``TimeFrame.halve`` when ``window.whole_days`` is ``True`` and a bound is near the
                ``datetime`` range.

        """
        current = self.sub_frame(frame)
        if window is None or current is None:
            return None
        halves = current.halve(whole_days=window.whole_days)
        if halves is None:
            return None
        left, _ = halves
        # nothing of the current sub-frame was yielded: same page_no, left half first, right half next
        return _Position(self.page_no, None, 0, (left.start, left.end, *self.bounds[1:]))

    def encode(self, frame: TimeFrame | None) -> str:
        """Return this position as a compact JSON cursor.

        The cursor holds the version, page number, paging cursor and row count, with sorted keys and no spaces.
        ``bounds`` is written, as ISO strings, only when it differs from the whole frame's two edges. The frame
        itself is not written, so the cursor does not record which frame it belongs to.

        Preconditions:
            None.

        Args:
            frame: The fetch's whole frame, or ``None``.

        Returns:
            A JSON object string that ``decode`` with the same ``frame`` turns back into an equal position.

        Raises:
            Nothing.

        """
        state: dict[str, Any] = {"v": CURSOR_VERSION, "n": self.page_no, "c": self.paging, "r": self.rows}
        if frame is not None and self.bounds != (frame.start, frame.end):  # unsplit: implied by the frame
            state["b"] = [b.isoformat() for b in self.bounds]
        return json.dumps(state, separators=(",", ":"), sort_keys=True)

    @classmethod
    def start(cls, frame: TimeFrame | None) -> _Position:
        """Return the position of a fetch's first page.

        Preconditions:
            None.

        Args:
            frame: The fetch's whole frame, or ``None``.

        Returns:
            Page 0 with no paging cursor and no rows, and ``bounds`` of the frame's two edges, or empty with no frame.

        Raises:
            Nothing.

        """
        return cls(0, None, 0, () if frame is None else (frame.start, frame.end))

    @classmethod
    def decode(cls, cursor: str, frame: TimeFrame | None) -> _Position:
        """Return the position a cursor from ``encode`` holds, checked against the fetch's frame.

        Without ``"b"`` the bounds are the whole frame's edges, so a cursor without sub-frames is accepted for any
        frame.

        Preconditions:
            None.

        Args:
            cursor: A cursor string, from outside the typed program (a stored resume cursor).
            frame: The fetch's whole frame, or ``None``.

        Returns:
            The decoded position.

        Raises:
            ValueError: When ``cursor`` is not JSON; is not an object with ``"v"`` equal to ``CURSOR_VERSION``; has a
                page number or row count that is not a non-negative ``int`` or a paging cursor that is neither a
                string nor ``null``; or from ``_bounds`` for sub-frames that are malformed or do not fit ``frame``.
            OverflowError: From ``_bounds`` (through ``ensure_utc``) when converting a sub-frame edge to UTC leaves the
                ``datetime`` range.

        """
        # our own saved cursor: a bad one is our bug, so ValueError
        try:
            state = json.loads(cursor)
        except ValueError:
            raise ValueError(f"not a fetch cursor: {cursor!r}") from None
        if not isinstance(state, dict) or state.get("v") != CURSOR_VERSION:
            raise ValueError(f"not a version {CURSOR_VERSION} fetch cursor: {cursor!r}")
        page_no, paging, rows = state.get("n"), state.get("c"), state.get("r")
        if not _count(page_no) or not _count(rows) or not (paging is None or isinstance(paging, str)):
            raise ValueError(f"malformed fetch cursor: {cursor!r}")
        return cls(page_no, paging, rows, _bounds(state.get("b"), frame, cursor))


def _count(value: object) -> TypeIs[int]:
    """Return whether a decoded JSON value is a non-negative integer, booleans excluded.

    Preconditions:
        None.

    Args:
        value: Any value, usually from ``json.loads``.

    Returns:
        ``True`` when ``value`` is an ``int`` other than a ``bool`` and is 0 or more.

    Raises:
        Nothing.

    """
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _bounds(raw: object, frame: TimeFrame | None, cursor: str) -> tuple[datetime, ...]:
    """Return the sub-frame edges a cursor holds, checked against the fetch's frame.

    Missing edges (``raw`` is ``None``) mean the whole frame. Edges must be strictly increasing, start no earlier
    than the frame's start and end exactly at its end. They need not lie on points the window can express; a
    whole-day window rejects such a sub-frame later, in ``Window.apply``.

    Preconditions:
        None.

    Args:
        raw: The cursor's ``"b"`` value from ``json.loads``, or ``None`` when absent.
        frame: The fetch's whole frame, or ``None``.
        cursor: The whole cursor, used only in error messages.

    Returns:
        The edges in UTC: empty with no frame, the frame's two edges when ``raw`` is ``None``, else the parsed list.

    Raises:
        ValueError: When ``raw`` is set but ``frame`` is ``None``; when ``raw`` is not a list of at least two strings;
            from ``datetime.fromisoformat`` for a string that is not an ISO datetime; from ``ensure_utc`` for a naive
            one; or when the edges are not increasing or do not lie in ``frame`` as described.
        OverflowError: From ``ensure_utc`` when converting an edge to UTC leaves the ``datetime`` range.

    """
    if frame is None:
        if raw is not None:
            raise ValueError(f"cursor has sub-frames but the fetch has no window: {cursor!r}")
        return ()
    if raw is None:
        return frame.start, frame.end
    if not isinstance(raw, list) or len(raw) < 2 or not all(isinstance(b, str) for b in raw):
        raise ValueError(f"malformed sub-frames in fetch cursor: {cursor!r}")
    bounds = tuple(ensure_utc(datetime.fromisoformat(b)) for b in raw)
    increasing = all(a < b for a, b in pairwise(bounds))
    if not increasing or bounds[0] < frame.start or bounds[-1] != frame.end:
        raise ValueError(f"cursor sub-frames do not lie in {frame}: {cursor!r}")
    return bounds


def _window(spec: EndpointSpec, name: str | None, frame: TimeFrame | None) -> Window | None:
    """Return the named window of a spec, checking that the window name and the frame are given together.

    Preconditions:
        None.

    Args:
        spec: The endpoint.
        name: The window name, or ``None``.
        frame: The fetch's whole frame, or ``None``.

    Returns:
        ``spec.windows[name]``, or ``None`` when both ``name`` and ``frame`` are ``None`` and the spec has no windows.

    Raises:
        ValueError: When exactly one of ``name`` and ``frame`` is ``None``; when neither is given but the spec has
            windows; or when the spec has no window called ``name``.

    """
    if (name is None) != (frame is None):
        raise ValueError(f"{spec.name}: a window and a time frame go together; got {name!r} and {frame}")
    if name is None:
        if spec.windows:
            raise ValueError(f"{spec.name}: needs a window, one of {sorted(spec.windows)}")
        return None
    try:
        return spec.windows[name]
    except KeyError:
        raise ValueError(f"{spec.name}: no window {name!r}, only {sorted(spec.windows)}") from None


def _split_up_front(
    spec: EndpointSpec, window: Window | None, frame: TimeFrame | None, at: _Position, body: Mapping[str, Any]
) -> _Position | None:
    """Return the split position when a sub-frame's first page reports more rows than the endpoint allows.

    Only a sub-frame's first page (``at.paging`` is ``None``) of an endpoint with both ``max_rows`` and
    ``total_at`` is checked. A total equal to ``max_rows`` is accepted.

    Preconditions:
        ``body`` is the response to the request built from ``at``, and the preconditions of ``_Position.split``.

    Args:
        spec: The endpoint.
        window: The fetch's window, or ``None``.
        frame: The fetch's whole frame, or ``None``.
        at: The position the response was requested at.
        body: The decoded response body.

    Returns:
        The position with the current sub-frame halved, or ``None`` when no split is needed.

    Raises:
        RowLimitReached: When the total exceeds ``max_rows`` and the sub-frame cannot be split, including when the
            fetch has no window.
        ClientErrorUnhandled: From ``EndpointSpec.total`` when ``total_at`` is set but the body has no count there.
        ValueError: From ``_Position.split``, only when its precondition is broken.
        TypeError: From ``_Position.split`` when ``window.whole_days`` is ``True`` and the zone is not a ``tzinfo``.
        OverflowError: From ``_Position.split`` when ``window.whole_days`` is ``True`` and a bound is near the
            ``datetime`` range.

    """
    # on a sub-frame's first page only: None when it fits, else the halved position
    if at.paging is not None or spec.max_rows is None:
        return None
    total = spec.total(body)
    if total is None or total <= spec.max_rows:
        return None
    halved = at.split(window, frame)
    if halved is None:
        raise RowLimitReached(endpoint=spec.name, limit=spec.max_rows)
    return halved


def _following(spec: EndpointSpec, at: _Position, count: int, paging: str | None) -> _Position | None:
    """Return the position of the page after the one just fetched, enforcing the row limit when there is no total.

    For an endpoint with ``max_rows`` but no ``total_at``, the page whose running row count reaches ``max_rows``
    raises ``RowLimitReached`` before it is yielded, whether or not more pages follow: with no total, rows at the cap
    may be a silent cut-off by the connector, which cannot be told apart from a frame of exactly ``max_rows`` rows. Such
    a frame is never split, because splitting after its rows were yielded would fetch them again.

    Preconditions:
        None.

    Args:
        spec: The endpoint.
        at: The position the page was requested at.
        count: The number of records on the page.
        paging: The paginator cursor of the next page in the sub-frame, or ``None`` when it was the sub-frame's last.

    Returns:
        ``at.advance`` with the running row count when ``paging`` is set, otherwise ``at.next_frame()``, which is
        ``None`` after the fetch's last page.

    Raises:
        RowLimitReached: When ``max_rows`` is set, ``total_at`` is not, and the running row count reaches
            ``max_rows``.

    """
    rows = at.rows + count
    # no total: rows at the cap may be a silent cut-off by the connector, so the frame fails rather than lose rows
    if spec.max_rows is not None and spec.total_at is None and rows >= spec.max_rows:
        raise RowLimitReached(endpoint=spec.name, limit=spec.max_rows)
    if paging is None:
        return at.next_frame()
    return at.advance(paging, rows)


async def fetch_pages[C](
    client: ConnectorClient[C],
    spec: EndpointSpec,
    window: str | None,  # a name in spec.windows; None for an endpoint without windows
    frame: TimeFrame | None,  # None exactly when window is None
    source_id: UUID,
    *,
    resume_from: str | None = None,  # cursor_next of the last saved page; None: from the start
    path_params: Mapping[str, str] = _NO_PATH_PARAMS,
    clock: Callable[[], datetime] = utc_now,
) -> AsyncIterator[Page]:
    """Fetch an endpoint's pages in order for a source, splitting the frame when a reported total exceeds the limit.

    Each sub-frame is paged to its end before the next starts. When a sub-frame's first page reports a total above
    ``spec.max_rows``, that response is dropped and the sub-frame is halved, repeatedly, until the total fits or the
    sub-frame cannot be split. Nothing is checked until the first iteration, because this is an async generator.
    The paging cursors sent are kept per sub-frame, so a paginator returning one already sent in the current
    sub-frame (a cycle such as A, B, A) fails instead of paging forever. They are reset at each sub-frame's first
    page, because a paginator's cursors (page numbers, offsets) legitimately repeat across sub-frames; a cycle that
    spans sub-frames is not one, and a resumed fetch knows only the cursors sent since it resumed.

    Preconditions:
        ``resume_from``, when given, is a ``cursor_in`` or ``cursor_next`` from a fetch with the same ``spec``,
        ``window``, ``frame``, ``source_id`` and ``path_params``. Only the frame's edges are checked, and only when the
        cursor holds sub-frames; a cursor from another frame without them is used as if it belonged to this one.

    Args:
        client: The client that sends each request.
        spec: The endpoint to fetch.
        window: The name of one of ``spec.windows``, or ``None`` when ``frame`` is ``None``.
        frame: The span of time to fetch, or ``None`` for an endpoint fetched without a window.
        source_id: The source the requests are made for.
        resume_from: A cursor to resume at, or ``None`` to start at page 0.
        path_params: Values for the fields in ``spec.path``. Defaults to none.
        clock: Returns the time recorded as each page's ``fetched_at``. Defaults to ``utc_now``.

    Yields:
        Each ``Page`` in order, numbered on from the resume cursor's page; the last has ``cursor_next`` of ``None``.

    Raises:
        ValueError: From ``_window`` for a window name and frame that do not match the spec; from
            ``EndpointSpec.request`` for wrong path parameters; from ``_Position.decode`` for a bad ``resume_from``;
            from ``Window.apply`` or ``Paginator.apply`` for a request they cannot build; or from ``Page`` when
            ``clock`` returns a naive datetime.
        OverflowError: From ``_Position.decode`` for a cursor edge, from ``Window.apply`` for a bound near the
            ``datetime`` range, or from ``_split_up_front`` for a whole-day split, when a ``datetime`` leaves its range.
        TypeError: From ``Window.apply`` or ``_split_up_front`` when a whole-day window or split meets a zone that is
            not a ``tzinfo``.
        RowLimitReached: From ``_split_up_front`` or ``_following`` when the endpoint's row limit is reached.
        ClientErrorUnhandled: From ``EndpointSpec.total``, ``EndpointSpec.record_count`` or
            ``Paginator.next_cursor`` for a body they cannot read; or from ``EndpointSpec.unhandled`` when the
            paginator returns a cursor already sent in the current sub-frame.
        Exception: Whatever ``client.send`` or ``clock`` raises, or whatever ``copy.deepcopy`` raises when
            ``Window.apply`` or ``Paginator.apply`` copies a request body.

    """
    win = _window(spec, window, frame)
    base = spec.request(path_params)
    at = _Position.start(frame) if resume_from is None else _Position.decode(resume_from, frame)
    cursor_in = resume_from
    sent: set[str] = set()  # paging cursors sent in the current sub-frame
    while True:
        if at.paging is None:  # a sub-frame's first page: its cursors start afresh
            sent.clear()
        else:
            sent.add(at.paging)
        sub = at.sub_frame(frame)
        request = base if win is None or sub is None else win.apply(base, sub)
        body = await client.send(spec.paginator.apply(request, at.paging), source_id)
        fetched_at = clock()
        halved = _split_up_front(spec, win, frame, at, body)
        if halved is not None:
            at = halved  # this page is dropped: nothing of the sub-frame is yielded before the split
            continue
        paging = spec.paginator.next_cursor(body, at.paging, spec=spec)
        if paging is not None and paging in sent:  # a cycle would page forever
            raise spec.unhandled(f"the API returned a cursor already used in this frame: {paging!r}")
        count = spec.record_count(body, last_page=paging is None)
        following = _following(spec, at, count, paging)
        cursor_next = None if following is None else following.encode(frame)
        yield Page(at.page_no, cursor_in, cursor_next, fetched_at, body, count)
        if following is None:
            return
        at, cursor_in = following, cursor_next
