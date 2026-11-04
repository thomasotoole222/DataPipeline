"""Tests for building and checking the frames of a model's targets."""

from typing import Annotated, ClassVar
from uuid import UUID

import pytest

from pipeline.checks.record import Int16, Int32, Keyed, Part, Record
from pipeline.checks.targets import ReplaceChildren, Rows, To, Upsert, target_of
from pipeline.db.quarantine import Failure
from pipeline.errors import Overflow
from pipeline.stages.transform import Transformed, frame_of, transform

SHOP = UUID("00000000-0000-0000-0000-000000000001")


class Line(Part, table="sales.line", writer=ReplaceChildren(key=("sku_id",))):
    """An order line; loads as child rows of ``Order``."""

    sku_id: Annotated[str, To("sku_id")]
    qty: Annotated[int, To("qty")]


class Order(Record, table="sales.order", writer=Upsert(("source_id", "order_id"))):
    """An order with its lines."""

    record_key: ClassVar[tuple[str, ...]] = ("order_id",)
    record_version: ClassVar[int] = 1

    order_id: Annotated[str, To("order_id")]
    lines: Annotated[list[Line], Rows()]


@pytest.mark.parametrize(("qty", "fits"), [(2**63 - 1, True), (-(2**63), True), (2**63, False), (-(2**63) - 1, False)])
def test_frame_of_keeps_ints_in_bigint_range(qty: int, fits: bool) -> None:
    """Check an int fits a bigint column only within its range, and one outside is refused without its value."""
    line = dict(target_of(Order).children)["lines"]
    row: dict[str, object] = {"source_id": str(SHOP), "order_id": "a", "sku_id": "s", "qty": qty}
    if fits:
        assert frame_of(line, [row])["qty"].to_list() == [qty]
    else:
        with pytest.raises(Overflow, match=r"^sales\.line\.qty: a value does not fit bigint NOT NULL$"):
            frame_of(line, [row])


def test_a_record_that_misfits_is_left_out_with_its_rows() -> None:
    """Check a record with a value that does not fit is a misfit, and the others keep their rows and owners."""
    accepted = [
        Keyed(Order(order_id="a", lines=[Line(sku_id="s1", qty=1)]), ("a",)),
        Keyed(Order(order_id="b", lines=[Line(sku_id="s2", qty=2**63)]), ("b",)),
        Keyed(Order(order_id="c", lines=[Line(sku_id="s3", qty=3), Line(sku_id="s4", qty=4)]), ("c",)),
    ]
    built = transform(Order, accepted, SHOP)
    (order, orders), (_, lines) = built.frames
    assert orders["order_id"].to_list() == ["a", "c"]
    assert lines["sku_id"].to_list() == ["s1", "s3", "s4"]
    assert built.owners == ((0, 2), (0, 2, 2))
    assert built.misfits == ((1, Failure("Overflow", 1, "sales.line.qty: a value does not fit bigint NOT NULL")),)
    with pytest.raises(ValueError, match=r"owners of sales\.order do not give one record per row"):
        Transformed(frames=built.frames, owners=((0,), (0, 2, 2)))
    assert order.table == "sales.order"


class Stat(Record, table="cipher.stat", writer=Upsert(("source_id", "day"))):
    """A day's stat with an integer date key and a smallint code."""

    record_key: ClassVar[tuple[str, ...]] = ("day",)
    record_version: ClassVar[int] = 1

    day: Annotated[Int32, To("day")]
    code: Annotated[Int16, To("code")]


@pytest.mark.parametrize(
    ("day", "code", "refused"),
    [
        (2**31 - 1, 2**15 - 1, None),
        (-(2**31), -(2**15), None),
        (2**31, 0, "day: a value does not fit integer NOT NULL"),
        (0, 2**15, "code: a value does not fit smallint NOT NULL"),
    ],
)
def test_frame_of_keeps_ints_in_their_columns_width(day: int, code: int, refused: str | None) -> None:
    """Check an int fits an integer or smallint column only within that width, as it does a bigint one."""
    row: dict[str, object] = {"source_id": str(SHOP), "day": day, "code": code}
    if refused is None:
        frame = frame_of(target_of(Stat), [row])
        assert (frame["day"].to_list(), frame["code"].to_list()) == ([day], [code])
    else:
        with pytest.raises(Overflow, match=rf"^cipher\.stat\.{refused}$"):
            frame_of(target_of(Stat), [row])
