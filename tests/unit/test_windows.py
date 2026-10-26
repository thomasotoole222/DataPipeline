"""Unit tests for the request windows in ``pipeline.connectors.windows``."""

from datetime import UTC, datetime

import pytest

from pipeline.connectors.spec import ConnectorRequest
from pipeline.connectors.windows import DateRangeWindow, FlatDateWindow, UnixSecondsWindow
from pipeline.timeutil import LONDON, TimeFrame


def test_unix_window_rejects_a_frame_inside_one_second() -> None:
    """Reject a frame that holds no whole second for ``UnixSecondsWindow``."""
    start = datetime(2026, 10, 1, 12, 0, 0, 100_000, tzinfo=UTC)
    frame = TimeFrame(start, start.replace(microsecond=900_000), LONDON)
    request = ConnectorRequest("POST", "/x", {}, None)
    with pytest.raises(ValueError, match="empty range of whole seconds"):
        UnixSecondsWindow("update_time_ge", "update_time_lt").apply(request, frame)


@pytest.mark.parametrize(("start", "end"), [("", "update_time_lt"), ("update_time_ge", ""), ("t", "t")])
def test_unix_window_rejects_empty_or_shared_names(start: str, end: str) -> None:
    """Reject empty or identical parameter names for ``UnixSecondsWindow``."""
    with pytest.raises(ValueError, match="non-empty and differ"):
        UnixSecondsWindow(start, end)


@pytest.mark.parametrize(("first", "last"), [("", "to"), ("from", ""), ("d", "d")])
def test_date_range_rejects_empty_or_shared_keys(first: str, last: str) -> None:
    """Reject empty or identical keys for ``DateRangeWindow``."""
    with pytest.raises(ValueError, match="non-empty and differ"):
        DateRangeWindow(("filters", "date_range"), end_inclusive=False, from_key=first, to_key=last)


@pytest.mark.parametrize(("first", "last"), [("", "time_to"), ("time_from", ""), ("d", "d")])
def test_flat_dates_rejects_empty_or_shared_names(first: str, last: str) -> None:
    """Reject empty or identical parameter names for ``FlatDateWindow``."""
    with pytest.raises(ValueError, match="non-empty and differ"):
        FlatDateWindow(first, last, end_inclusive=True)
