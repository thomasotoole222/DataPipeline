"""Unit tests for a Rows child's ``Severity.DROP`` lookup, a lookup's ``source_column``, and ReplaceChildren scope."""

from typing import Annotated, Any, ClassVar

import pytest
from pydantic import Field

from pipeline.checks.checks import Lookup, Rule, Severity, Validation
from pipeline.checks.record import Part, Record
from pipeline.checks.targets import ReplaceChildren, Rows, To, Upsert, target_of
from pipeline.stages.load import _scope
from pipeline.stages.validate import _drop_rows


class Stock(Part, table="cipher.stock", writer=Upsert(("warehouse",))):
    """One warehouse's stock of a product."""

    warehouse: Annotated[str, To("warehouse")]
    available: Annotated[int, To("available")]


class Product(Record, container=True):
    """A product's stock across warehouses, with a one-to-one stock row too."""

    record_key: ClassVar[tuple[str, ...]] = ("product",)
    record_version: ClassVar[int] = 1
    product: Annotated[str, To("product_id")]
    stock: Annotated[list[Stock], Rows()] = Field(default_factory=list)


class Single(Record, container=True):
    """A product with one stock row."""

    record_key: ClassVar[tuple[str, ...]] = ("product",)
    record_version: ClassVar[int] = 1
    product: Annotated[str, To("product_id")]
    stock: Annotated[Stock | None, Rows()] = None


SELLER_ONLY = Lookup[Stock](
    "seller_warehouse_only",
    lambda row: [row.warehouse],
    table="cipher.dim_fulfillment",
    column="tiktok_warehouse_id",
    severity=Severity.DROP,
    rows="stock",
)


def test_a_drop_lookup_on_a_list_rows_field_is_accepted() -> None:
    """Accept a drop lookup naming a list Rows field of the model."""
    assert Validation(model=Product, checks=(SELLER_ONLY,)).checks == (SELLER_ONLY,)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"severity": Severity.DROP}, "exactly when it is drop"),
        ({"severity": Severity.ERROR, "rows": "stock"}, "exactly when it is drop"),
        ({"severity": Severity.DROP, "rows": ""}, "exactly when it is drop"),
        ({"severity": Severity.DROP, "rows": "stock", "until": "warehouses"}, "until is set exactly when"),
    ],
    ids=["drop without rows", "rows without drop", "empty rows", "drop with until"],
)
def test_a_drop_lookup_refuses_a_bad_declaration(kwargs: dict[str, Any], match: str) -> None:
    """Refuse rows without drop, drop without rows, an empty rows and an until on a drop."""
    with pytest.raises(ValueError, match=match):
        Lookup("x", lambda row: [], table="t.t", column="c", **kwargs)


@pytest.mark.parametrize("source_column", ["", "Shop_Id", "shop id"])
def test_a_lookup_refuses_a_bad_source_column(source_column: str) -> None:
    """Refuse a source column that is not a lower-case bare column name."""
    with pytest.raises(ValueError, match="source_column must be a lower-case name"):
        Lookup("x", lambda row: [], table="t.t", column="c", until="j", source_column=source_column)


def test_a_lookup_takes_a_source_column() -> None:
    """Accept a lower-case source column; it defaults to None."""
    assert Lookup("x", lambda row: [], table="t.t", column="c", until="j", source_column="shop_id").source_column == (
        "shop_id"
    )
    assert Lookup("x", lambda row: [], table="t.t", column="c", until="j").source_column is None


def test_a_rule_cannot_drop() -> None:
    """Refuse drop on a Rule: only a lookup on a child's rows drops."""
    with pytest.raises(ValueError, match="drop is for a Lookup"):
        Rule("x", lambda record, ctx: True, severity=Severity.DROP)


@pytest.mark.parametrize(
    ("model", "rows", "match"),
    [
        (Product, "product", "which is not a Rows field"),
        (Product, "nope", "which is not a Rows field"),
        (Single, "stock", "a one-to-one child, not a list"),
    ],
)
def test_a_validation_refuses_a_drop_lookup_on_no_list_rows_field(model: type[Record], rows: str, match: str) -> None:
    """Refuse a drop lookup whose rows is not a Rows field of the model, or is a one-to-one child."""
    lookup = Lookup[Stock]("x", lambda row: [row.warehouse], table="t.t", column="c", severity=Severity.DROP, rows=rows)
    with pytest.raises(ValueError, match=match):
        Validation(model=model, checks=(lookup,))


def test_drop_rows_keeps_only_the_rows_found_and_counts_the_rest() -> None:
    """Remove the rows whose value is not found, in a copy, and count them; a record with none missing is kept."""
    record = Product.model_validate(
        {"product": "p1", "stock": [{"warehouse": "w1", "available": 3}, {"warehouse": "fbt", "available": 9}]}
    )
    pruned, cut = _drop_rows(record, (SELLER_ONLY,), {"seller_warehouse_only": frozenset({"w1"})})
    assert [row.warehouse for row in pruned.stock] == ["w1"]
    assert [row.warehouse for row in record.stock] == ["w1", "fbt"]
    assert cut == [(SELLER_ONLY, 1)]
    same, none = _drop_rows(record, (SELLER_ONLY,), {"seller_warehouse_only": frozenset({"w1", "fbt"})})
    assert same is record
    assert none == []


class Sku(Part, table="cipher.promo_sku", writer=ReplaceChildren(("sku",))):
    """A SKU replaced under its promotion product."""

    sku: Annotated[str, To("sku")]


class Price(Part, table="cipher.promo_price", writer=Upsert(())):
    """A promotion product's one price, upserted under it."""

    amount: Annotated[int, To("amount")]


class PromotionProduct(Part, table="cipher.promo_product", writer=ReplaceChildren(("product",))):
    """A product replaced under its promotion, with SKUs and a price."""

    product: Annotated[str, To("product")]
    skus: Annotated[list[Sku], Rows()] = Field(default_factory=list)
    price: Annotated[Price | None, Rows()] = None


class Promotion(Record, table="cipher.promotion", writer=Upsert(("source_id", "promotion"))):
    """A promotion with ReplaceChildren products."""

    record_key: ClassVar[tuple[str, ...]] = ("promotion",)
    record_version: ClassVar[int] = 1
    promotion: Annotated[str, To("promotion")]
    products: Annotated[list[PromotionProduct], Rows()] = Field(default_factory=list)


def test_a_replace_children_grandchild_is_scoped_as_its_replace_children_parent() -> None:
    """Scope a ReplaceChildren grandchild by its ReplaceChildren parent's scope, and any other child by its parent."""
    root = target_of(Promotion)
    products = dict(root.children)["products"]
    skus, price = dict(products.children)["skus"], dict(products.children)["price"]
    scopes = {products: _scope(root, products, "staged_promotion", {})}
    assert scopes[products] == (("source_id", "promotion"), "staged_promotion")
    assert _scope(products, skus, "staged_product", scopes) == (("source_id", "promotion"), "staged_promotion")
    assert _scope(products, price, "staged_product", scopes) == (
        ("source_id", "promotion", "product"),
        "staged_product",
    )
