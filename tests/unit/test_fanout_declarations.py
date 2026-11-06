"""Unit tests for fan-out declarations: ``IdParam``, ``EndpointSpec``'s id rules, the id sources and ``BatchJob``."""

from datetime import UTC, date, datetime
from typing import Annotated, Any, ClassVar
from uuid import UUID
from zoneinfo import ZoneInfo

import pytest
from opentelemetry.sdk.metrics import MeterProvider

from pipeline.checks.checks import DETAIL_RULE, Rule, Validation
from pipeline.checks.record import Int32, Record, TaskDay
from pipeline.checks.targets import ReplacePartition, To, Upsert
from pipeline.connectors.fanout import FromList, FromQuery, id_text
from pipeline.connectors.paging import CursorPaginator, NoPaging
from pipeline.connectors.spec import ConnectorRequest, EndpointSpec, IdParam, IdPlace
from pipeline.connectors.windows import UnixSecondsWindow
from pipeline.db.raw import RawPage
from pipeline.errors import ClientErrorUnhandled, IdRefused
from pipeline.jobs.base import BatchJob
from pipeline.telemetry import Instruments

CURSOR = CursorPaginator(
    token_param="page_token",  # noqa: S106 - a parameter name, not a password
    size_param="page_size",
    size=10,
    next_at=("data", "next_page_token"),
)
WINDOW = {"update": UnixSecondsWindow(start_param="update_time_ge", end_param="update_time_lt")}
SEARCH = EndpointSpec(
    "tiktok_shop.products_search",
    "tiktok_shop",
    "POST",
    "/product/202502/products/search",
    CURSOR,
    ("data", "products"),
    windows=WINDOW,
)
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


def spec(path: str = "/items", method: Any = "POST", **kwargs: Any) -> EndpointSpec:
    """Build a test endpoint on the ``tiktok_shop`` connector."""
    return EndpointSpec(
        "tiktok_shop.items", "tiktok_shop", method, path, kwargs.pop("paginator", CURSOR), ("d",), **kwargs
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"place": "path", "name": "x"}, "a path field, a query parameter or the body"),
        ({"place": IdPlace.PATH, "name": ""}, "needs a name"),
        ({"place": IdPlace.BODY, "name": "ids", "batch": 0}, "1 or more"),
        ({"place": IdPlace.BODY, "name": "ids", "batch": True}, "1 or more"),
        ({"place": IdPlace.PATH, "name": "x", "batch": 2}, "one per call"),
        ({"place": IdPlace.QUERY, "name": "x", "batch": 2}, "one per call"),
        ({"place": IdPlace.PATH, "name": "x", "inject": ""}, "injected id field"),
        ({"place": IdPlace.BODY, "name": "ids", "batch": 2, "inject": "id"}, "injected id field"),
        ({"place": IdPlace.PATH, "name": "x", "encoding": "csv"}, "only a query id takes an encoding"),
        ({"place": IdPlace.BODY, "name": "ids", "encoding": "json"}, "only a query id takes an encoding"),
        ({"place": IdPlace.QUERY, "name": "x", "encoding": "xml"}, "only a query id takes an encoding, csv or json"),
        ({"place": IdPlace.QUERY, "name": "x", "batch": 2, "wrap": ("ids",)}, "one per call"),
        ({"place": IdPlace.QUERY, "name": "x", "encoding": "csv", "wrap": ("ids",)}, "only json-encoded ids"),
        ({"place": IdPlace.QUERY, "name": "x", "encoding": "json", "wrap": ()}, "key path"),
        ({"place": IdPlace.QUERY, "name": "x", "encoding": "json", "wrap": ("a", "")}, "key path"),
        ({"place": IdPlace.QUERY, "name": "x", "batch": 2, "encoding": "csv", "inject": "id"}, "injected id field"),
        ({"place": IdPlace.BODY, "name": "ids", "batch": 2, "id_at": ()}, "key path"),
        ({"place": IdPlace.BODY, "name": "ids", "batch": 2, "id_at": ("data", "")}, "key path"),
    ],
)
def test_id_param_refuses_a_bad_declaration(kwargs: dict[str, Any], match: str) -> None:
    """Refuse an empty name, a bad batch, encoding, wrap or id_at, and an injected field with a batch."""
    with pytest.raises(ValueError, match=match):
        IdParam(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"path": "/items/{x}", "id_param": IdParam(place=IdPlace.PATH, name="y")}, "not just a path id's own"),
        ({"path": "/s/{s}/i/{x}", "id_param": IdParam(place=IdPlace.PATH, name="x")}, "not just a path id's own"),
        ({"path": "/items/{x}", "id_param": IdParam(place=IdPlace.QUERY, name="x")}, "not just a path id's own"),
        ({"fixed_query": {"x": "1"}, "id_param": IdParam(place=IdPlace.QUERY, name="x")}, "fixed or paging query"),
        ({"method": "GET", "id_param": IdParam(place=IdPlace.BODY, name="ids")}, "body of a GET"),
        ({"fixed_body": {"ids": []}, "id_param": IdParam(place=IdPlace.BODY, name="ids")}, "fixed or paging body"),
        ({"id_param": IdParam(place=IdPlace.QUERY, name="page_token")}, "fixed or paging query"),
        ({"id_param": {"place": "path"}}, "must be an IdParam"),
    ],
)
def test_a_spec_refuses_ids_it_cannot_take(kwargs: dict[str, Any], match: str) -> None:
    """Refuse an id placement that clashes with the endpoint's path, method or other parameters."""
    with pytest.raises(ValueError, match=match):
        spec(**kwargs)


def test_request_for_puts_one_id_in_the_path_encoded() -> None:
    """Substitute a path id, percent-encoded."""
    assert GET.request_for(["a/b"]).path == "/product/202309/products/a%2Fb"


@pytest.mark.parametrize(("value", "match"), [("..", "dot path values"), (".", "dot path values"), ("", "empty")])
def test_request_for_refuses_a_path_id_it_cannot_place_as_id_refused(value: str, match: str) -> None:
    """Refuse a path id the request builder cannot take as IdRefused, a holdable error, not a ValueError."""
    with pytest.raises(IdRefused, match=match) as caught:
        GET.request_for([value])
    assert not isinstance(caught.value, ValueError)
    assert caught.value.__cause__ is None


def test_request_for_puts_one_id_in_the_query() -> None:
    """Add a query id beside the fixed query."""
    query = spec(method="GET", fixed_query={"v": "2"}, id_param=IdParam(place=IdPlace.QUERY, name="live_id"))
    assert query.request_for(["7"]).query == {"v": "2", "live_id": "7"}


@pytest.mark.parametrize(
    ("id_param", "value"),
    [
        (IdParam(place=IdPlace.QUERY, name="ids", batch=3, encoding="csv"), "a,b"),
        (IdParam(place=IdPlace.QUERY, name="ids", batch=3, encoding="json"), '["a","b"]'),
        (
            IdParam(place=IdPlace.QUERY, name="ids", batch=100, encoding="json", wrap=("filtering", "campaign_ids")),
            '{"filtering":{"campaign_ids":["a","b"]}}',
        ),
    ],
    ids=["csv", "json", "json wrapped twice"],
)
def test_request_for_encodes_a_batch_of_ids_in_the_query(id_param: IdParam, value: str) -> None:
    """Join a query batch by commas, or send it as a compact JSON array, nested in its wrap keys."""
    query = spec(method="GET", fixed_query={"v": "2"}, id_param=id_param)
    assert query.request_for(["a", "b"]).query == {"v": "2", "ids": value}


def test_request_for_wraps_json_ids_for_a_gmv_max_filter() -> None:
    """Send filtering={"campaign_ids":[...]}, as the GMV Max report takes its campaign ids."""
    where = IdParam(place=IdPlace.QUERY, name="filtering", batch=100, encoding="json", wrap=("campaign_ids",))
    request = spec(method="GET", id_param=where).request_for(["1", "2"])
    assert request.query == {"filtering": '{"campaign_ids":["1","2"]}'}
    assert spec(method="GET", id_param=where).request_for(["1"]).query == {"filtering": '{"campaign_ids":["1"]}'}


def test_request_for_refuses_a_comma_in_a_comma_joined_id() -> None:
    """Refuse, as a holdable IdRefused, a csv id holding a comma, which would be read as two ids."""
    batched = spec(method="GET", id_param=IdParam(place=IdPlace.QUERY, name="ids", batch=2, encoding="csv"))
    with pytest.raises(IdRefused, match="holds a comma"):
        batched.request_for(["a", "b,c"])


def test_request_for_puts_a_batch_of_ids_in_the_body() -> None:
    """Set the batch as a list at the top-level body key, beside the fixed body."""
    batched = spec(fixed_body={"k": 1}, id_param=IdParam(place=IdPlace.BODY, name="ids", batch=3))
    assert batched.request_for(["a", "b"]).body == {"k": 1, "ids": ["a", "b"]}


@pytest.mark.parametrize("ids", [[], ["a", "b"]])
def test_request_for_refuses_too_few_or_too_many_ids(ids: list[str]) -> None:
    """Refuse a call for no ids, or for more than the batch."""
    with pytest.raises(ValueError, match="a call is for 1 to 1 ids"):
        GET.request_for(ids)


def test_request_for_refuses_a_spec_without_ids() -> None:
    """Refuse ids for an endpoint that takes none."""
    with pytest.raises(ValueError, match="takes no ids"):
        SEARCH.request_for(["a"])


@pytest.mark.parametrize(
    ("body", "records"),
    [
        ({"data": {"id": "p1"}}, [{"id": "p1"}]),
        ({"data": None}, []),
        ({"code": 0}, []),
    ],
)
def test_one_record_reads_an_object_or_nothing(body: dict[str, Any], records: list[Any]) -> None:
    """Read one record object at records_at, and nothing or null there as no record."""
    assert GET.records(body) == records
    assert GET.record_count(body) == len(records)


def test_one_record_refuses_a_list() -> None:
    """Refuse a value that is neither an object nor null."""
    with pytest.raises(ClientErrorUnhandled, match="not an object"):
        GET.records({"data": [{"id": "p1"}]})


@pytest.mark.parametrize(
    ("value", "text"),
    [("p1", "p1"), (7, "7"), ("", None), (None, None), (True, None), (1.5, None), ({"id": 1}, None)],
)
def test_id_text_takes_a_non_empty_string_or_an_integer(value: object, text: str | None) -> None:
    """Write a string as it is and an integer in decimal; refuse anything else."""
    assert id_text(value) == text


def test_a_list_endpoint_cannot_itself_fan_out() -> None:
    """Refuse a list endpoint with an id_param: no chaining."""
    with pytest.raises(ValueError, match="cannot chain"):
        FromList(spec=GET, id_at=("id",))


def test_a_list_needs_a_key_path() -> None:
    """Refuse an empty id_at."""
    with pytest.raises(ValueError, match="key path"):
        FromList(spec=SEARCH, id_at=())


@pytest.mark.parametrize(
    ("record", "match"),
    [([1], "not an object"), ({"x": 1}, "no id at id"), ({"id": None}, "not a string or an integer")],
)
def test_a_list_record_without_a_readable_id_is_unhandled(record: object, match: str) -> None:
    """Report a list record with no readable id as a response the spec cannot read."""
    with pytest.raises(ClientErrorUnhandled, match=match):
        FromList(spec=SEARCH, id_at=("id",)).id_of(record)


def test_a_list_reads_an_id_at_its_key_path() -> None:
    """Read a nested id, an integer as text."""
    assert FromList(spec=SEARCH, id_at=("product", "id")).id_of({"product": {"id": 12}}) == "12"


@pytest.mark.parametrize(
    ("sql", "match"),
    [
        ("  ", "needs SQL"),
        ("SELECT id FROM t WHERE s = %(source)s", r"not \['source'\]"),
        ("SELECT id FROM t WHERE s = %s", r"not '%s' at index 27"),
        ("SELECT id FROM t WHERE name LIKE 'a%'", r"not \"%'\" at index 35"),
        ("SELECT id FROM t WHERE s = %(source_id)d", r"not '%\(source_id\)d'"),
        ("SELECT id FROM t WHERE s = %(frame-start)s", r"not \['frame-start'\]"),
        ("SELECT id FROM t WHERE s = %(source_id)", r"not '%\(source_id\)'"),
        ("SELECT id FROM t WHERE x = 1 %", r"not '%' at index 29"),
    ],
)
def test_an_id_query_refuses_blank_sql_and_unknown_parameters(sql: Any, match: str) -> None:
    """Refuse a blank query, a parameter outside QUERY_PARAMS, and any % sequence but %% and %(name)s."""
    with pytest.raises(ValueError, match=match):
        FromQuery(sql)


def test_an_id_query_takes_its_named_parameters_and_literal_percents() -> None:
    """Accept every bound parameter, and a doubled % as a literal."""
    FromQuery(
        "SELECT id FROM t WHERE s = %(source_id)s AND k = %(partition_key)s AND a >= %(frame_start)s "
        "AND a < %(frame_end)s AND name LIKE 'x%%'"
    )


def test_detail_failed_is_reserved() -> None:
    """Refuse a declared check named detail_failed."""
    with pytest.raises(ValueError, match="other than"):
        Rule(DETAIL_RULE, lambda record, ctx: True)


class Product(Record, table="t_fan.product", writer=Upsert(("source_id", "product_id"))):
    """A product, from the list row or the per-id call."""

    record_key: ClassVar[tuple[str, ...]] = ("id",)
    record_version: ClassVar[int] = 1

    id: Annotated[str, To("product_id")]


class Client:
    """A client that is never called."""

    connector = "tiktok_shop"
    hashed_keys = frozenset({"data"})

    async def send(self, request: ConnectorRequest, source_id: UUID) -> dict[str, Any]:
        """Fail: no test here sends."""
        raise AssertionError("not sent")


def job(**attributes: Any) -> BatchJob[Product]:
    """Build a fanned-out products job, with ``attributes`` overriding its declaration."""
    declared: dict[str, Any] = {
        "name": "products",
        "spec": GET,
        "window": "update",
        "partition": "day",
        "validation": Validation(model=Product),
        "id_source": FromList(spec=SEARCH, id_at=("id",), keep_list_row=True),
    }
    declared.update(attributes)
    runner = type("Products", (BatchJob,), declared)
    meter = MeterProvider().get_meter("test")
    instruments = Instruments(
        task_duration=meter.create_histogram("pipeline.task.duration", unit="s"),
        rows=meter.create_counter("pipeline.rows"),
        connector_requests=meter.create_counter("pipeline.connector.requests"),
        log_dropped=meter.create_counter("pipeline.log.dropped"),
        slots_busy=meter.create_up_down_counter("pipeline.worker.slots_busy"),
        queue_wait=meter.create_histogram("pipeline.task.queue_wait", unit="s"),
        heartbeat_lag=meter.create_histogram("pipeline.lease.heartbeat_lag", unit="s"),
        lease_losses=meter.create_counter("pipeline.lease.losses"),
        lease_unwind=meter.create_histogram("pipeline.lease.unwind", unit="s"),
        unknown_jobs=meter.create_counter("pipeline.task.unknown_job"),
        worker_paused=meter.create_counter("pipeline.worker.paused"),
        check_failures=meter.create_counter("pipeline.check.failures"),
        text_replaced=meter.create_counter("pipeline.connector.text_replaced"),
    )
    return runner(client=Client(), zone=ZoneInfo("UTC"), instruments=instruments)  # type: ignore[no-any-return]


def test_a_fanned_out_job_is_accepted() -> None:
    """Accept a list into a per-id spec without windows, the window applying to the list."""
    job()


def test_a_query_sourced_job_needs_no_windowed_spec() -> None:
    """Accept a day partition whose frame only the query reads."""
    job(id_source=FromQuery("SELECT 1 WHERE %(frame_start)s IS NOT NULL"))


@pytest.mark.parametrize(
    ("attributes", "match"),
    [
        ({"id_source": None}, "needs both"),
        ({"spec": SEARCH, "window": "update"}, "needs both"),
        ({"id_source": "SELECT 1"}, "FromList or a FromQuery"),
        ({"concurrency": 0}, "concurrency"),
        ({"concurrency": True}, "concurrency"),
        ({"window": "create"}, "has no window 'create'"),
        (
            {
                "id_source": FromList(
                    spec=EndpointSpec("cruva.samples", "cruva", "POST", "/samples", CURSOR, ("d",)), id_at=("id",)
                )
            },
            "another connector",
        ),
        (
            {
                "spec": spec(id_param=IdParam(place=IdPlace.BODY, name="ids", batch=2)),
                "window": "update",
            },
            "keep_list_row needs one id per call",
        ),
    ],
)
def test_a_fanned_out_job_refuses_a_bad_declaration(attributes: dict[str, Any], match: str) -> None:
    """Refuse a fan-out missing a half, with a bad source or concurrency, a foreign list, or a window it lacks."""
    with pytest.raises(ValueError, match=match):
        job(**attributes)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"ids": ()}, "at least one id"),
        ({"ids": ("a", None)}, "no None"),
        ({"error_class": "PermanentError"}, "needs the ids"),
        ({"ids": ("a",), "error_class": ""}, "must not be empty"),
    ],
)
def test_a_raw_page_refuses_empty_ids_and_an_error_without_ids(kwargs: dict[str, Any], match: str) -> None:
    """Refuse an empty ids list, and a failed call's error class with no ids."""
    with pytest.raises(ValueError, match=match):
        RawPage(
            task_id=1,
            page_no=0,
            connector="tiktok_shop",
            endpoint="product",
            cursor_in=None,
            cursor_next=None,
            fetched_at=datetime(2026, 10, 19, tzinfo=UTC),
            payload={},
            **kwargs,
        )


def test_a_raw_page_holds_its_ids_as_a_tuple() -> None:
    """Turn the ids a database row gives as a list into a tuple."""
    page = RawPage(
        task_id=1,
        page_no=0,
        connector="tiktok_shop",
        endpoint="product",
        cursor_in=None,
        cursor_next=None,
        fetched_at=datetime(2026, 10, 19, tzinfo=UTC),
        payload={"error": "x"},
        ids=["a", "b"],  # type: ignore[arg-type]  # as class_row passes a text[]
        error_class="PermanentError",
    )
    assert page.ids == ("a", "b")


class Daily(Record, table="t_fan.daily", writer=ReplacePartition("day")):
    """A product's day, replaced per partition."""

    record_key: ClassVar[tuple[str, ...]] = ("id",)
    record_version: ClassVar[int] = 1

    id: Annotated[str, To("product_id")]
    day: Annotated[date, To("day")]


def test_a_fanned_out_job_refuses_a_replace_partition_target() -> None:
    """Refuse ReplacePartition in a fan-out, whose empty id source would leave the partition as it was."""
    with pytest.raises(ValueError, match=r"t_fan\.daily replaces a partition, which a fan-out cannot"):
        job(validation=Validation(model=Daily))


def test_a_none_fan_out_with_a_windowed_endpoint_says_it_needs_a_day_partition() -> None:
    """Refuse a windowed list under a none partition, saying a day partition is needed."""
    with pytest.raises(ValueError, match="products_search has windows, so it needs a day partition, not none"):
        job(partition="none", window=None)


class ProductDay(Record, table="t_fan.product_day", writer=Upsert(("source_id", "product_id", "date_key"))):
    """A product's day, dated by the task's day."""

    record_key: ClassVar[tuple[str, ...]] = ("id", "day")
    record_version: ClassVar[int] = 1

    id: Annotated[str, To("product_id")]
    day: Annotated[Int32, TaskDay(), To("date_key")]


def test_a_task_day_is_accepted_under_a_day_partition() -> None:
    """Accept a model with a TaskDay field in a job with a day partition."""
    job(validation=Validation(model=ProductDay))


def test_a_task_day_is_refused_under_a_none_partition() -> None:
    """Refuse a model with a TaskDay field in a job with a none partition, which has no day to fill it with."""
    with pytest.raises(ValueError, match="day is a TaskDay, so it needs a day partition, not none"):
        job(validation=Validation(model=ProductDay), id_source=FromQuery("SELECT 1"), partition="none", window=None)
