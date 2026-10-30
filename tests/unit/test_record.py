"""Unit tests for ``pipeline.checks.record``: the boolean field type ``Bool`` and its table ``BOOL_VALUES``."""

from decimal import Decimal
from typing import ClassVar

import pytest
from pydantic import ValidationError

from pipeline.checks.record import BOOL_VALUES, Bool, Record


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
