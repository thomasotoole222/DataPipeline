"""Unit tests for ``pipeline.connectors.fetch.fetch_pages`` and ``map_error`` on a "no record" answer."""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID
from zoneinfo import ZoneInfo

import httpx
import pytest

from pipeline.connectors.client import ErrorOutcome, ErrorRule, _result, map_error
from pipeline.connectors.fetch import fetch_pages
from pipeline.connectors.paging import CursorPaginator, NoPaging
from pipeline.connectors.spec import ConnectorRequest, EndpointSpec, IdParam, IdPlace, Place
from pipeline.connectors.windows import UnixSecondsWindow
from pipeline.errors import ConnectorResponseError, NoRecord, Outcome, PermanentError, classify
from pipeline.timeutil import TimeFrame

SOURCE = UUID("00000000-0000-0000-0000-000000000001")
AT = datetime(2026, 11, 4, 12, tzinfo=UTC)
GET = EndpointSpec(
    "tiktok_shop.product",
    "tiktok_shop",
    "GET",
    "/product/202309/products/{product_id}",
    NoPaging(),
    ("data",),
    id_param=IdParam(place=IdPlace.PATH, name="product_id"),
    one_record=True,
)
PAGED = EndpointSpec(
    "tiktok_shop.minutes",
    "tiktok_shop",
    "GET",
    "/minutes",
    CursorPaginator(token_param="page_token", size_param="page_size", size=2, next_at=("data", "next")),  # noqa: S106
    ("data", "minutes"),
    windows={"update": UnixSecondsWindow(start_param="ge", end_param="lt", place=Place.QUERY)},
    max_rows=2,
    total_at=("data", "total"),
)


class Client:
    """A client that answers each request with the next of ``answers``; an exception is raised instead."""

    connector = "tiktok_shop"
    hashed_keys = frozenset({"data"})

    def __init__(self, *answers: dict[str, Any] | Exception) -> None:
        """Answer with ``answers`` in turn."""
        self.answers = list(answers)
        self.requests: list[ConnectorRequest] = []

    async def send(self, request: ConnectorRequest, source_id: UUID) -> dict[str, Any]:
        """Record the request and give the next answer."""
        self.requests.append(request)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def error(code: int) -> ConnectorResponseError:
    """Return a TikTok error response with ``code``."""
    return ConnectorResponseError(
        connector="tiktok_shop", path="/p", status=200, code=code, message="not found", request_id="r", headers={}
    )


def test_a_no_record_rule_maps_to_no_record_which_is_permanent_and_labelled() -> None:
    """Map an error a NO_RECORD rule matches to NoRecord, a PermanentError, recorded as no_record."""
    found = map_error([ErrorRule(outcome=ErrorOutcome.NO_RECORD, code=28001006)], error(28001006))
    assert isinstance(found, NoRecord)
    assert isinstance(found, PermanentError)
    assert classify(found) == (Outcome.FAILED, "NoRecord")
    assert _result(found) == "no_record"
    assert isinstance(map_error([ErrorRule(outcome=ErrorOutcome.NO_RECORD, code=1)], error(2)), Exception)


@pytest.mark.parametrize("body", [{"data": {}}, {"data": None}, {"code": 0}])
def test_an_empty_object_is_no_record_for_a_one_record_endpoint(body: dict[str, Any]) -> None:
    """Read an empty object at records_at, like nothing or null there, as no record."""
    assert GET.records(body) == []
    assert GET.record_count(body) == 0


async def test_a_no_record_answer_lands_as_one_empty_last_page() -> None:
    """Turn a first request answered with NoRecord into one empty last page."""
    client = Client(NoRecord("frozen"))
    pages = [page async for page in fetch_pages(client, GET, None, None, SOURCE, ids=["p1"], clock=lambda: AT)]
    assert [(p.page_no, p.cursor_in, p.cursor_next, p.payload, p.record_count) for p in pages] == [
        (0, None, None, {}, 0)
    ]


async def test_a_no_record_answer_after_the_first_page_propagates() -> None:
    """Refuse NoRecord on a later page of a sub-frame, where an empty page would cut it short."""
    frame = TimeFrame(datetime(2026, 11, 4, tzinfo=UTC), datetime(2026, 11, 5, tzinfo=UTC), ZoneInfo("UTC"))
    first = {"data": {"minutes": [1, 2], "next": "t2", "total": 2}}
    client = Client(first, NoRecord("gone"))
    with pytest.raises(NoRecord):
        [page async for page in fetch_pages(client, PAGED, "update", frame, SOURCE, clock=lambda: AT)]


async def test_a_no_record_sub_frame_is_skipped_and_the_next_one_fetched() -> None:
    """Skip a split sub-frame answered with NoRecord, taking no page number, and fetch the next sub-frame."""
    frame = TimeFrame(datetime(2026, 11, 4, tzinfo=UTC), datetime(2026, 11, 5, tzinfo=UTC), ZoneInfo("UTC"))
    too_many = {"data": {"minutes": [1, 2], "next": "", "total": 3}}
    right = {"data": {"minutes": [3], "next": "", "total": 1}}
    client = Client(too_many, NoRecord("left half"), right)
    pages = [page async for page in fetch_pages(client, PAGED, "update", frame, SOURCE, clock=lambda: AT)]
    assert [(p.page_no, p.payload, p.cursor_next) for p in pages] == [(0, right, None)]
    start = int(frame.start.timestamp())
    assert [r.query["ge"] for r in client.requests] == [str(start), str(start), str(start + 12 * 3600)]


def test_a_rule_can_name_no_record_by_status_too() -> None:
    """Accept a NO_RECORD rule on a status, as on any condition."""
    response = ConnectorResponseError(
        connector="c", path="/p", status=404, code=None, message="x", request_id=None, headers=httpx.Headers()
    )
    assert isinstance(map_error([ErrorRule(outcome=ErrorOutcome.NO_RECORD, status=404)], response), NoRecord)
