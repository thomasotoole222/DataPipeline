"""Unit tests for the paginators in ``cipher_pipeline.sources.paging``."""

from typing import Any

import pytest

from cipher_pipeline.errors import ClientErrorUnhandled
from cipher_pipeline.sources.paging import CursorPaginator, HasMore, PageNumberPaginator, ShortPage, TotalCount
from cipher_pipeline.sources.spec import EndpointSpec

SAMPLES_PAGING = PageNumberPaginator("page", "page_size", 2, HasMore(("data", "has_more")))
SAMPLES = EndpointSpec("cruva.samples", "cruva", "POST", "/samples", SAMPLES_PAGING, ("data", "results"))
ADS_PAGING = PageNumberPaginator("page", "page_size", 2, TotalCount(("data", "total_count")))
ADS = EndpointSpec("cruva.ads_creatives", "cruva", "POST", "/ads/creatives", ADS_PAGING, ("data", "creatives"))


@pytest.mark.parametrize("data", [{"has_more": False}, {"has_more": False, "results": None}])
def test_has_more_last_page_may_leave_out_or_null_its_records(data: dict[str, Any]) -> None:
    """End ``HasMore`` paging on a false flag when the last page leaves out or nulls its records."""
    assert SAMPLES_PAGING.next_cursor({"data": data}, None, spec=SAMPLES) is None


@pytest.mark.parametrize("data", [{"has_more": True}, {"has_more": True, "results": None}])
def test_has_more_true_without_records_raises_unhandled(data: dict[str, Any]) -> None:
    """Reject a missing or ``null`` record list when ``HasMore`` says more pages follow."""
    with pytest.raises(ClientErrorUnhandled, match=r"records at data\.results"):
        SAMPLES_PAGING.next_cursor({"data": data}, None, spec=SAMPLES)


def test_total_count_last_page_may_leave_out_its_records() -> None:
    """End ``TotalCount`` paging when the last page leaves out its records."""
    assert ADS_PAGING.next_cursor({"data": {"total_count": 3}}, "2", spec=ADS) is None


def test_total_count_missing_records_before_total_raises_unhandled() -> None:
    """Reject a missing record list on a page before the total is reached."""
    with pytest.raises(ClientErrorUnhandled, match=r"no records at data\.creatives"):
        ADS_PAGING.next_cursor({"data": {"total_count": 5}}, None, spec=ADS)


@pytest.mark.parametrize(("token", "size"), [("", "page_size"), ("page_token", ""), ("page", "page")])
def test_cursor_rejects_empty_or_shared_names(token: str, size: str) -> None:
    """Refuse to build a ``CursorPaginator`` with an empty or shared parameter name."""
    with pytest.raises(ValueError, match="non-empty and differ"):
        CursorPaginator(token, size, 100, ("data", "next_page_token"))


@pytest.mark.parametrize(("page", "size"), [("", "page_size"), ("page", ""), ("page", "page")])
def test_page_number_rejects_empty_or_shared_names(page: str, size: str) -> None:
    """Refuse to build a ``PageNumberPaginator`` with an empty or shared parameter name."""
    with pytest.raises(ValueError, match="non-empty and differ"):
        PageNumberPaginator(page, size, 100, ShortPage())
