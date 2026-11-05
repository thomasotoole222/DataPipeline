"""Tests for building and checking the frames of a model's targets."""

from typing import Annotated, ClassVar
from uuid import UUID

import pytest
from pydantic import Field

from pipeline.checks.record import Int16, Int32, Keyed, Part, Record
from pipeline.checks.targets import ReplaceChildren, Rows, To, Upsert, Versioned, target_of
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


class Component(Part, table="cipher.bundle", writer=ReplaceChildren(("component",))):
    """A bundle component under its SKU."""

    component: Annotated[str, To("component")]


class Price(Part, table="cipher.price", writer=Versioned((), valid_from="f", valid_to="t")):
    """A SKU's one-to-one price."""

    price: Annotated[int, To("price")]


class Sku(Part, table="cipher.sku", writer=Upsert(("sku_id",))):
    """A SKU under its product."""

    sku_id: Annotated[str, To("sku_id")]
    price: Annotated[Price | None, Rows()] = None
    parts: Annotated[list[Component], Rows()] = Field(default_factory=list)


class Product(Record, table="cipher.product", writer=Upsert(("source_id", "product_id"))):
    """A product with its SKUs."""

    record_key: ClassVar[tuple[str, ...]] = ("product_id",)
    record_version: ClassVar[int] = 1

    product_id: Annotated[str, To("product_id")]
    skus: Annotated[list[Sku], Rows()] = Field(default_factory=list)


def test_nested_children_load_parents_first_with_one_to_one_and_none() -> None:
    """Check frames come parents first at every depth, a single part gives one row, and None gives none."""
    accepted = [
        Keyed(
            Product(
                product_id="p1",
                skus=[
                    Sku(
                        sku_id="s1", price=Price(price=5), parts=[Component(component="c1"), Component(component="c2")]
                    ),
                    Sku(sku_id="s2"),
                ],
            ),
            ("p1",),
        ),
        Keyed(Product(product_id="p2"), ("p2",)),
    ]
    built = transform(Product, accepted, SHOP)
    tables = [target.table for target, _ in built.frames]
    assert tables == ["cipher.product", "cipher.sku", "cipher.price", "cipher.bundle"]
    rows = {target.table: frame.rows() for target, frame in built.frames}
    assert rows == {
        "cipher.product": [(str(SHOP), "p1"), (str(SHOP), "p2")],
        "cipher.sku": [("p1", "s1"), ("p1", "s2")],  # the product's key, without the source column
        "cipher.price": [("s1", 5)],  # s2 has no price
        "cipher.bundle": [("s1", "c1"), ("s1", "c2")],
    }
    assert built.owners == ((0, 1), (0, 0), (0,), (0, 0))


def test_a_child_without_its_parent_is_an_orphan() -> None:
    """Check Transformed refuses a keyed child whose parent is not in the frames."""
    built = transform(Product, [], SHOP)
    with pytest.raises(ValueError, match=r"cipher\.sku has no parent in frames"):
        Transformed(frames=built.frames[1:2], owners=((),))


class Traffic(Part, table="cipher.traffic", writer=ReplaceChildren(("channel",))):
    """A product-day's views by channel."""

    channel: Annotated[str, To("channel")]
    views: Annotated[int, To("views")]


class ProductDay(Record, container=True):
    """A product-day with no table of its own."""

    record_key: ClassVar[tuple[str, ...]] = ("product_id", "day")
    record_version: ClassVar[int] = 1

    product_id: Annotated[str, To("product_id")]
    day: Annotated[Int32, To("date_key")]
    traffic: Annotated[list[Traffic], Rows()] = Field(default_factory=list)


def test_a_container_frame_holds_only_its_key_and_comes_first() -> None:
    """Check a container's frame is its source and key columns, before its children's frames."""
    accepted = [
        Keyed(ProductDay(product_id="p1", day=20261104, traffic=[Traffic(channel="live", views=9)]), ("p1", 20261104))
    ]
    built = transform(ProductDay, accepted, SHOP)
    (container, keys), (traffic, rows) = built.frames
    assert container.table is None
    assert keys.rows() == [(str(SHOP), "p1", 20261104)]
    assert traffic.table == "cipher.traffic"
    assert rows.rows() == [(str(SHOP), "p1", 20261104, "live", 9)]
