"""Tests for building and checking the frames of a model's targets."""

from typing import Annotated, ClassVar
from uuid import UUID

import pytest

from pipeline.checks.record import Part, Record
from pipeline.checks.targets import ReplaceChildren, Rows, To, Upsert, target_of
from pipeline.stages.transform import frame_of

SHOP = UUID("00000000-0000-0000-0000-000000000001")


class Line(Part, table="sales.line", writer=ReplaceChildren()):
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
    """Check an int fits a bigint column only within its range, and one outside is refused, naming it."""
    line = dict(target_of(Order).children)["lines"]
    row: dict[str, object] = {"source_id": str(SHOP), "order_id": "a", "sku_id": "s", "qty": qty}
    if fits:
        assert frame_of(line, [row])["qty"].to_list() == [qty]
    else:
        with pytest.raises(ValueError, match=rf"line\.qty: {qty} does not fit bigint NOT NULL$"):
            frame_of(line, [row])
