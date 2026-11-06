"""Unit tests for ``pipeline.checks.record``: ``Bool`` and its ``BOOL_VALUES``, ``DateKey`` and ``TaskDay``."""

from collections.abc import Mapping
from decimal import Decimal
from typing import Annotated, Any, ClassVar

import pytest
from pydantic import Field, ValidationError

from pipeline.checks.checks import RESHAPE_RULE, Lookup, Rule, Severity
from pipeline.checks.record import (
    BOOL_VALUES,
    Bool,
    DateKey,
    Int32,
    Part,
    Record,
    TaskDay,
    key_of,
    task_day_fields,
)
from pipeline.checks.targets import To, Upsert, target_of
from pipeline.stages.validate import _day_key, _Landed, _reshaped


class Flagged(Record):
    """A record with a required and an optional boolean field."""

    record_key: ClassVar = ("id",)
    record_version: ClassVar = 1
    id: str
    flag: Bool
    maybe: Bool | None = None


@pytest.mark.parametrize(("value", "expected"), [*BOOL_VALUES.items(), (1, True), (0, False)])
def test_bool_maps_every_value_in_the_table(value: object, expected: bool) -> None:
    """Every key of ``BOOL_VALUES``, then 1 and 0, parses to its bool through ``model_validate`` in Python mode."""
    assert Flagged.model_validate({"id": "a", "flag": value}).flag is expected


@pytest.mark.parametrize(("value", "expected"), [(" Yes ", True), ("OFF", False)])
def test_bool_trims_and_lower_cases_text(value: str, expected: bool) -> None:
    """Text is trimmed and lower-cased before the lookup."""
    assert Flagged.model_validate({"id": "a", "flag": value}).flag is expected


@pytest.mark.parametrize(
    "value", ["maybe", 2, 1.0, Decimal(1), None], ids=["unknown text", "two", "float", "decimal", "none"]
)
def test_bool_refuses_any_other_value(value: object) -> None:
    """A value that is not a key of ``BOOL_VALUES`` fails the strict bool check at its field, with no value shown."""
    with pytest.raises(ValidationError) as info:
        Flagged.model_validate({"id": "a", "flag": value})
    (error,) = info.value.errors()
    assert error["loc"] == ("flag",)
    assert error["type"] == "bool_type"
    assert error["msg"] == "Input should be a valid boolean"


def test_true_and_false_are_also_one_and_zero() -> None:
    """1 and 0 hash and compare equal to True and False, so they are those keys, not keys of their own."""
    assert BOOL_VALUES[1] is True
    assert BOOL_VALUES[0] is False
    assert len(BOOL_VALUES) == 14


def test_an_optional_bool_takes_none() -> None:
    """A field typed ``Bool | None`` takes None as None and still maps text."""
    assert Flagged.model_validate({"id": "a", "flag": True, "maybe": None}).maybe is None
    assert Flagged.model_validate({"id": "a", "flag": True, "maybe": "no"}).maybe is False


class ShopDay(Record, table="cipher.shop_day", writer=Upsert(("source_id", "date_key"))):
    """A shop's day, keyed by the task's day."""

    record_key: ClassVar[tuple[str, ...]] = ("date_key",)
    record_version: ClassVar[int] = 1
    date_key: Annotated[Int32, TaskDay(), To("date_key")]
    gmv: Annotated[int | None, To("gmv")] = None


def test_a_task_day_field_may_be_a_key_field_and_loads_into_integer() -> None:
    """Accept TaskDay on an Int32 key field: it is listed, read as the key, and loads into an integer column."""
    assert task_day_fields(ShopDay) == ("date_key",)
    assert key_of(ShopDay, {"date_key": 20261104}) == (20261104,)
    assert [(c.name, c.postgres) for c in target_of(ShopDay).columns] == [
        ("source_id", "uuid"),
        ("date_key", "integer"),
        ("gmv", "bigint"),
    ]


def test_a_model_without_task_day_lists_none() -> None:
    """List no field for a model with no TaskDay mark."""
    assert task_day_fields(Flagged) == ()


@pytest.mark.parametrize(
    ("annotation", "message"),
    [
        (Annotated[str, TaskDay()], "must be an int, not <class 'str'>"),
        (Annotated[int | None, TaskDay()], "must be an int"),
        (Annotated[int, TaskDay(), TaskDay()], "takes one TaskDay mark"),
        (Annotated[int, TaskDay(), Field(alias="day")], "takes no alias"),
        (Annotated[int, TaskDay(), Field(validation_alias="day")], "takes no alias"),
    ],
    ids=["text", "optional", "two marks", "alias", "validation alias"],
)
def test_task_day_refuses_a_field_it_cannot_fill(annotation: Any, message: str) -> None:
    """Refuse TaskDay on a field that is not a plain int, carries two marks, or is read under an alias."""
    with pytest.raises(TypeError, match=message):
        type(
            "Bad",
            (Record,),
            {"__annotations__": {"day": annotation}, "record_key": ("day",), "record_version": 1},
        )


def test_task_day_refuses_a_nested_parts_field() -> None:
    """Refuse TaskDay on a Part that is not a Record: only a record's own payload is filled."""
    with pytest.raises(TypeError, match="TaskDay fills a Record's own field, not a nested part's"):

        class Inner(Part):
            day: Annotated[int, TaskDay()]


@pytest.mark.parametrize(("key", "day"), [("2026-11-04", 20261104), ("0001-01-01", 10101)])
def test_the_task_day_is_the_partition_key_as_yyyymmdd(key: str, day: int) -> None:
    """Turn a day partition's YYYY-MM-DD key into the YYYYMMDD int a TaskDay field gets."""
    assert _day_key(key) == day


@pytest.mark.parametrize("key", ["20261104", "none", "2026-02-30"])
def test_the_task_day_refuses_a_key_that_is_not_a_day(key: str) -> None:
    """Refuse a partition key that is not a YYYY-MM-DD date, the compact ISO form included."""
    with pytest.raises(ValueError, match=r"YYYY-MM-DD|Invalid isoformat|must be in range"):
        _day_key(key)


class VideoDay(Record, table="cipher.video_day", writer=Upsert(("source_id", "video_id", "date_key"))):
    """A video's day, keyed by the report's ISO date."""

    record_key: ClassVar[tuple[str, ...]] = ("video_id", "start_date")
    record_version: ClassVar[int] = 1
    video_id: Annotated[str, To("video_id")]
    start_date: Annotated[DateKey, To("date_key")]
    end_date: Annotated[DateKey | None, To("end_date_key")] = None


@pytest.mark.parametrize(("value", "key"), [("2026-11-04", 20261104), (20261104, 20261104), ("0001-01-01", 10101)])
def test_a_date_key_reads_an_iso_date_or_a_yyyymmdd_int(value: object, key: int) -> None:
    """Read a YYYY-MM-DD date, or an int already in YYYYMMDD, as the YYYYMMDD int, in the record and in its key."""
    assert VideoDay.model_validate({"video_id": "v", "start_date": value}).start_date == key
    assert key_of(VideoDay, {"video_id": "v", "start_date": value}) == ("v", key)


@pytest.mark.parametrize(
    "value",
    ["20261104", "2026-02-30", "2026-11-04T00:00:00", 20260230, 0, -20261104, True, 20261104.0, None],
    ids=["compact text", "no such day", "a datetime", "no such int day", "zero", "negative", "bool", "float", "null"],
)
def test_a_date_key_refuses_anything_but_a_real_day(value: object) -> None:
    """Refuse text not written YYYY-MM-DD, digits that name no day, and any other type, at the field."""
    with pytest.raises(ValidationError) as info:
        VideoDay.model_validate({"video_id": "v", "start_date": value})
    assert [error["loc"] for error in info.value.errors()] == [("start_date",)]


def test_a_date_key_loads_into_integer_and_may_be_optional() -> None:
    """Map a DateKey to an integer column, and an optional one to a nullable integer that takes None."""
    columns = {c.name: (c.postgres, c.nullable) for c in target_of(VideoDay).columns}
    assert columns["date_key"] == ("integer", False)
    assert columns["end_date_key"] == ("integer", True)
    assert VideoDay.model_validate({"video_id": "v", "start_date": 20261104, "end_date": None}).end_date is None


def flatten(record: Mapping[str, Any]) -> Mapping[str, Any]:
    """Lift each product's skus into one list."""
    return {**record, "skus": [sku for product in record["products"] for sku in product["skus"]]}


class Promotion(Record):
    """A promotion whose skus are lifted out of its products before it is parsed."""

    record_key: ClassVar[tuple[str, ...]] = ("id",)
    record_version: ClassVar[int] = 1
    reshape = flatten
    id: str
    skus: list[dict[str, Any]]


class LaterPromotion(Promotion):
    """A subclass, which inherits the reshape."""


def test_a_reshape_is_a_class_attribute_that_a_subclass_inherits() -> None:
    """Keep the declared reshape on the class, inherited by a subclass; a Record without one has none."""
    assert Promotion.reshape is flatten
    assert LaterPromotion.reshape is flatten
    assert not hasattr(Flagged, "reshape")
    assert flatten({"id": "a", "products": [{"skus": [1]}, {"skus": [2]}]})["skus"] == [1, 2]


def test_a_reshape_that_is_not_callable_is_refused() -> None:
    """Refuse a reshape that is not a function."""
    with pytest.raises(TypeError, match="reshape must be a function of one record, not 'flatten'"):
        type("Bad", (Record,), {"__annotations__": {"id": str}, "record_key": ("id",), "record_version": 1,
                                "reshape": "flatten"})  # fmt: skip


def test_reshape_failed_is_reserved() -> None:
    """Refuse a declared rule or lookup named reshape_failed."""
    with pytest.raises(ValueError, match="other than"):
        Rule(RESHAPE_RULE, lambda record, ctx: True)
    with pytest.raises(ValueError, match="other than"):
        Lookup(RESHAPE_RULE, lambda record: [], table="t.t", column="c", severity=Severity.ERROR)


def odd(record: Mapping[str, Any]) -> Any:
    """Return a list for a record marked ``list``, raise for one marked ``raise``, else the record with a flag."""
    if record.get("list"):
        return [record]
    if record.get("raise"):
        raise ValueError("no blocks\x00here")
    return {**record, "shaped": True}


class Odd(Record):
    """A record whose reshape fails in both ways."""

    record_key: ClassVar[tuple[str, ...]] = ("id",)
    record_version: ClassVar[int] = 1
    reshape = odd
    id: str


def test_reshaped_keeps_the_payload_and_lists_each_failure() -> None:
    """Reshape a mapping, keep a non-mapping unshaped, and list a raise or a non-mapping result as failures."""
    landed = [_Landed(0, 0, {"id": "a"}), _Landed(0, 1, [1]), _Landed(0, 2, {"raise": 1}), _Landed(0, 3, {"list": 1})]
    kept, failed = _reshaped(Odd, landed)
    assert [(item.index, item.payload, item.shaped, item.record) for item in kept] == [
        (0, {"id": "a"}, {"id": "a", "shaped": True}, {"id": "a", "shaped": True}),
        (1, [1], None, [1]),
    ]
    assert [(item.index, message) for item, message in failed] == [
        (2, "ValueError: no blocks\ufffdhere"),
        (3, "reshape returned a list, not a mapping"),
    ]


def test_reshaped_without_a_reshape_keeps_every_record() -> None:
    """Keep every record unshaped for a model with no reshape."""
    landed = [_Landed(0, 0, {"id": "a"})]
    assert _reshaped(Flagged, landed) == (landed, [])
