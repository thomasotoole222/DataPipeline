"""Tests for declaring a model's target table with class keywords."""

from datetime import date
from decimal import Decimal
from typing import Annotated, ClassVar

import pytest
from annotated_types import Ge, Gt, Le, Lt
from pydantic import Field

from pipeline.checks.record import Int16, Int32, Part, Record
from pipeline.checks.targets import (
    ReplaceChildren,
    ReplacePartition,
    Rows,
    Snapshot,
    To,
    Update,
    Upsert,
    Versioned,
    Width,
    Writer,
    mismatches,
    row_key,
    target_of,
    walk,
)
from pipeline.db.catalog import TableColumn, TableInfo


class Line(Part, table="sales.line", writer=ReplaceChildren(key=("sku_id",))):
    """An order line; loads as child rows of ``Order``."""

    sku_id: Annotated[str, To("sku_id")]


class Order(Record, table="sales.order", writer=Upsert(("source_id", "order_id"))):
    """An order with its lines."""

    record_key: ClassVar[tuple[str, ...]] = ("order_id",)
    record_version: ClassVar[int] = 1

    order_id: Annotated[str, To("order_id")]
    lines: Annotated[list[Line], Rows()]


class Loop(Part):
    """A part that contains itself."""

    name: str
    child: Loop | None = None


def test_source_column_defaults() -> None:
    """Check a Record's source column defaults to source_id, a Part has none, and a Record can opt out with None."""

    class Unsourced(Record, table="sales.unsourced", writer=Upsert(("order_id",)), source_column=None):
        record_key: ClassVar[tuple[str, ...]] = ("order_id",)
        record_version: ClassVar[int] = 1

        order_id: Annotated[str, To("order_id")]

    assert target_of(Order).source_column == "source_id"
    assert Line.__target__ is not None
    assert Line.__target__.source_column is None
    assert target_of(Unsourced).source_column is None


def test_subclass_does_not_inherit_table() -> None:
    """Check a subclass that gives no keywords declares no table."""

    class Copy(Order): ...

    assert Copy.__target__ is None
    with pytest.raises(TypeError, match=r"^Copy declares no table$"):
        target_of(Copy)


def test_models_without_table_still_validate() -> None:
    """Check models that declare no table still build and validate."""

    class Note(Part):
        text: str

    class Plain(Record):
        record_key: ClassVar[tuple[str, ...]] = ("id",)
        record_version: ClassVar[int] = 1

        id: str
        note: Note

    plain = Plain.model_validate({"id": "a", "note": {"text": "t"}, "new": 1})
    assert Plain.__target__ is None
    assert plain.id == "a"
    assert plain.note.text == "t"


def test_record_checks_still_run() -> None:
    """Check a Record with a table still has its record_key checked."""
    with pytest.raises(TypeError, match=r"^Keyless must set record_key$"):

        class Keyless(Record, table="sales.keyless", writer=Upsert(("source_id", "order_id"))):
            record_version: ClassVar[int] = 1

            order_id: Annotated[str, To("order_id")]


@pytest.mark.parametrize(
    ("table", "writer"),
    [("sales.half", None), (None, Upsert(("source_id", "order_id")))],
    ids=["table without writer", "writer without table"],
)
def test_declaration_refused(table: str | None, writer: Writer | None) -> None:
    """Check table and writer are given together."""
    with pytest.raises(TypeError, match=r"^Half: table and writer are given together$"):

        class Half(Record, table=table, writer=writer):
            record_key: ClassVar[tuple[str, ...]] = ("order_id",)
            record_version: ClassVar[int] = 1

            order_id: Annotated[str, To("order_id")]


@pytest.mark.parametrize(
    ("keywords", "field", "error", "message"),
    [
        (
            "table='t.x', writer=ReplacePartition('d')",
            "d: Annotated[date | None, To('d')] = None",
            ValueError,
            "ReplacePartition column 'd' cannot allow None",
        ),
        (
            "table='t.x', writer=Upsert(key=('source_id',))",
            "loop: Loop | None = None",
            TypeError,
            "Loop contains itself",
        ),
    ],
    ids=["nullable partition column", "part contains itself"],
)
def test_target_refused(keywords: str, field: str, error: type[Exception], message: str) -> None:
    """Check a declaration whose target does not build is refused when the class is created."""
    code = (
        f"class Refused(Record, {keywords}):\n"
        "    record_key: ClassVar[tuple[str, ...]] = ()\n"
        "    record_version: ClassVar[int] = 1\n"
        f"    {field}\n"
    )
    names = {
        "Annotated": Annotated,
        "ClassVar": ClassVar,
        "date": date,
        "Loop": Loop,
        "Record": Record,
        "ReplacePartition": ReplacePartition,
        "To": To,
        "Upsert": Upsert,
    }
    with pytest.raises(error, match=message):
        exec(code, names)  # noqa: S102


def test_record_table_key_accepted() -> None:
    """Check a writer key of the source column plus the record_key columns is accepted, in any order."""

    class Stock(Record, table="sales.stock", writer=Snapshot(("sku", "warehouse", "source_id"))):
        record_key: ClassVar[tuple[str, ...]] = ("warehouse_id", "sku_id")
        record_version: ClassVar[int] = 1

        warehouse_id: Annotated[str, To("warehouse")]
        sku_id: Annotated[str, To("sku")]
        qty: Annotated[int, To("qty")]

    assert target_of(Stock).writer == Snapshot(("sku", "warehouse", "source_id"))


def test_record_table_key_refused() -> None:
    """Check a key field without a To, and a writer key other than source column plus key columns, are refused."""
    with pytest.raises(TypeError, match=r"^Untold\.order_id is in record_key, so it needs a To$"):

        class Untold(Record, table="sales.untold", writer=Upsert(("source_id", "code"))):
            record_key: ClassVar[tuple[str, ...]] = ("order_id",)
            record_version: ClassVar[int] = 1

            order_id: str
            code: Annotated[str, To("code")]

    with pytest.raises(TypeError, match=r"^Wide: the writer's key must be the columns order_id, source_id$"):

        class Wide(Record, table="sales.wide", writer=Upsert(("source_id", "order_id", "code"))):
            record_key: ClassVar[tuple[str, ...]] = ("order_id",)
            record_version: ClassVar[int] = 1

            order_id: Annotated[str, To("order_id")]
            code: Annotated[str, To("code")]


def test_child_row_key_is_inherited_then_own() -> None:
    """Check a child's row key is the parent's writer key followed by its own key, and a root's is its writer key."""
    root = target_of(Order)
    assert row_key(root) == ("source_id", "order_id")
    assert row_key(dict(root.children)["lines"]) == ("source_id", "order_id", "sku_id")


@pytest.mark.parametrize(
    ("key", "field", "message"),
    [
        ((), "sku_id: Annotated[str, To('sku_id')]", "the writer's key must name distinct columns"),
        (("qty",), "sku_id: Annotated[str, To('sku_id')]", "names 'qty', which is not a target column"),
        (("sku_id",), "sku_id: Annotated[str | None, To('sku_id')] = None", "key column 'sku_id' cannot allow None"),
    ],
    ids=["empty key", "unknown column", "nullable column"],
)
def test_replace_children_key_refused(key: tuple[str, ...], field: str, message: str) -> None:
    """Check a ReplaceChildren key must name the child's own non-nullable columns."""
    code = f"class Child(Part, table='sales.child', writer=ReplaceChildren(key={key!r})):\n    {field}\n"
    with pytest.raises(ValueError, match=message):
        exec(code, {"Annotated": Annotated, "Part": Part, "ReplaceChildren": ReplaceChildren, "To": To})  # noqa: S102


def test_mismatches_needs_a_unique_key_on_the_child_row_key() -> None:
    """Check a child table without a unique key on its row key is refused."""
    line = dict(target_of(Order).children)["lines"]
    columns = {
        name: TableColumn(name=name, type="text", nullable=False, has_default=False, writable=True)
        for name in ("source_id", "order_id", "sku_id")
    }
    columns["source_id"] = TableColumn(name="source_id", type="uuid", nullable=False, has_default=False, writable=True)
    parent_only = TableInfo(table="sales.line", columns=columns, unique_keys=frozenset({frozenset({"order_id"})}))
    assert mismatches(line, parent_only) == ["sales.line has no unique key on (source_id, order_id, sku_id)"]
    full = TableInfo(
        table="sales.line", columns=columns, unique_keys=frozenset({frozenset({"source_id", "order_id", "sku_id"})})
    )
    assert mismatches(line, full) == []


VERSIONED = Versioned(("source_id", "sku"), valid_from="valid_from_key", valid_to="valid_to_key", opens_from="created")


class Sku(Record, table="cipher.dim_sku", writer=VERSIONED):
    """A versioned SKU: versions keyed by its id and first day."""

    record_key: ClassVar[tuple[str, ...]] = ("id",)
    record_version: ClassVar[int] = 1

    id: Annotated[str, To("sku")]
    title: Annotated[str | None, To("title")] = None
    created: Annotated[int | None, To("created")] = None


def _sku_table(**changes: TableColumn | None) -> TableInfo:
    """Build a fitting dim_sku catalog entry, then apply ``changes`` (a column, or None to drop it)."""

    def col(name: str, kind: str, nullable: bool, writable: bool = True, default: bool = False) -> TableColumn:
        return TableColumn(name=name, type=kind, nullable=nullable, has_default=default, writable=writable)

    columns = {
        "source_id": col("source_id", "uuid", False),
        "sku": col("sku", "text", False),
        "title": col("title", "text", True),
        "created": col("created", "bigint", True),
        "valid_from_key": col("valid_from_key", "integer", False),
        "valid_to_key": col("valid_to_key", "integer", True),
        "is_current": col("is_current", "boolean", True, writable=False, default=True),
    }
    for name, column in changes.items():
        if column is None:
            columns.pop(name)
        else:
            columns[name] = column
    return TableInfo(
        table="cipher.dim_sku",
        columns=columns,
        unique_keys=frozenset({frozenset({"source_id", "sku", "valid_from_key"})}),
    )


def test_versioned_is_a_snapshot_with_the_snapshot_key_rules() -> None:
    """Check a Versioned target keeps its writer, counts as a Snapshot, and has the key as its row key."""
    target = target_of(Sku)
    assert target.writer == VERSIONED
    assert isinstance(target.writer, Snapshot)
    assert row_key(target) == ("source_id", "sku")


@pytest.mark.parametrize(
    ("writer", "field", "message"),
    [
        ("Versioned(('source_id', 'sku'), valid_from='d', valid_to='d')", "", "two distinct lower-case column names"),
        ("Versioned(('source_id', 'sku'), valid_from='valid_from_key', valid_to='e')",
         "f: Annotated[int, To('valid_from_key')]", "sets valid_from_key itself"),
        ("Versioned(('source_id', 'sku'), valid_from='d', valid_to='e', opens_from='nope')", "",
         "opens_from must be a target column"),
        ("Versioned(('source_id', 'sku'), valid_from='d', valid_to='e', opens_from='name')",
         "name: Annotated[str, To('name')]", "opens_from must be a target column"),
        ("Versioned(('sku',), valid_from='d', valid_to='e')", "", "must include the source column 'source_id'"),
        ("Versioned(('source_id', 'sku'), valid_from='d', valid_to='e', prune=False, allow_empty=True)", "",
         "allow_empty lets an empty frame prune every row, so it needs prune"),
    ],
)  # fmt: skip
def test_versioned_declarations_refused(writer: str, field: str, message: str) -> None:
    """Check a Versioned writer refuses equal day columns, a field loading into one, a bad opens_from, no source.

    It also refuses allow_empty without prune, which would have no effect.
    """
    code = (
        f"class Bad(Record, table='cipher.bad', writer={writer}):\n"
        "    record_key: ClassVar[tuple[str, ...]] = ('id',)\n"
        "    record_version: ClassVar[int] = 1\n"
        "    id: Annotated[str, To('sku')]\n"
        f"    {field}\n"
    )
    names = {"Annotated": Annotated, "ClassVar": ClassVar, "Record": Record, "To": To, "Versioned": Versioned}
    with pytest.raises(ValueError, match=message):
        exec(code, names)  # noqa: S102


def test_a_top_level_snapshot_or_versioned_may_turn_its_prune_off() -> None:
    """Check prune defaults to on, and a top-level Snapshot or Versioned target keeps prune=False on its writer."""

    class Window(
        Record, table="cipher.window", writer=Versioned(("source_id", "kit"), valid_from="f", valid_to="t", prune=False)
    ):
        record_key: ClassVar[tuple[str, ...]] = ("id",)
        record_version: ClassVar[int] = 1

        id: Annotated[str, To("kit")]

    class Stock(Record, table="cipher.stock", writer=Snapshot(("source_id", "sku"), prune=False)):
        record_key: ClassVar[tuple[str, ...]] = ("sku",)
        record_version: ClassVar[int] = 1

        sku: Annotated[str, To("sku")]

    assert Snapshot(("sku",)).prune
    assert not Snapshot(("sku",)).allow_empty
    assert target_of(Window).writer == Versioned(("source_id", "kit"), valid_from="f", valid_to="t", prune=False)
    assert target_of(Stock).writer == Snapshot(("source_id", "sku"), prune=False)


def test_versioned_accepts_rows_children() -> None:
    """Check a Versioned table may own Rows children, which inherit its key without the source column."""

    class Kit(Record, table="cipher.kit", writer=Versioned(("source_id", "kit"), valid_from="f", valid_to="t")):
        record_key: ClassVar[tuple[str, ...]] = ("id",)
        record_version: ClassVar[int] = 1

        id: Annotated[str, To("kit")]
        lines: Annotated[list[Line], Rows()]

    line = dict(target_of(Kit).children)["lines"]
    assert line.inherited == ("source_id", "kit")  # a ReplaceChildren child inherits the whole writer key
    assert row_key(line) == ("source_id", "kit", "sku_id")


def test_versioned_table_that_fits() -> None:
    """Check a fitting table: integer day columns, a version key, no removed_at, valid_from left to the writer."""
    assert mismatches(target_of(Sku), _sku_table()) == []


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"valid_from_key": None}, "cipher.dim_sku.valid_from_key does not exist"),
        ({"valid_to_key": TableColumn(name="valid_to_key", type="integer", nullable=False, has_default=False,
                                      writable=True)}, "valid_to_key must be nullable"),
        ({"valid_to_key": TableColumn(name="valid_to_key", type="text", nullable=True, has_default=False,
                                      writable=True)}, "valid_to_key is text, a Versioned writer needs"),
        ({"valid_from_key": TableColumn(name="valid_from_key", type="date", nullable=False, has_default=False,
                                        writable=True)}, "must both be date keys or both dates"),
    ],
)  # fmt: skip
def test_versioned_table_mismatches(changes: dict[str, TableColumn | None], message: str) -> None:
    """Check missing, nullable, mistyped and mixed day columns are reported."""
    assert any(message in found for found in mismatches(target_of(Sku), _sku_table(**changes)))


@pytest.mark.parametrize(
    "unique",
    [
        frozenset({"source_id", "sku"}),  # one row per id: versions need their first day
        frozenset({"sku", "valid_from_key"}),  # narrower: refuses two sources' versions of one id on a day
        frozenset({"valid_from_key"}),  # narrower still: refuses two new ids opened on one day
        frozenset({"source_id", "sku", "title", "valid_from_key"}),  # wider: lets two versions of a key share a day
    ],
)
def test_versioned_table_needs_a_unique_key_on_exactly_its_versions(unique: frozenset[str]) -> None:
    """Check a table is refused unless a unique key is on exactly the writer key and valid_from."""
    info = _sku_table()
    other = TableInfo(table=info.table, columns=info.columns, unique_keys=frozenset({unique}))
    assert mismatches(target_of(Sku), other) == [
        "cipher.dim_sku has no unique key on exactly sku, source_id, valid_from_key"
    ]


def test_int32_and_int16_load_into_integer_and_smallint() -> None:
    """Check Int32 and Int16 fields map to integer and smallint, nullable under None, a plain int staying bigint."""

    class Day(Record, table="cipher.fact_day", writer=Upsert(("source_id", "date_key"))):
        record_key: ClassVar[tuple[str, ...]] = ("day",)
        record_version: ClassVar[int] = 1

        day: Annotated[Int32, To("date_key")]
        status: Annotated[Int16 | None, To("status")] = None
        orders: Annotated[int, To("orders")] = 0

    found = {column.name: (column.postgres, column.nullable) for column in target_of(Day).columns}
    assert found == {
        "source_id": ("uuid", False),
        "date_key": ("integer", False),
        "status": ("smallint", True),
        "orders": ("bigint", False),
    }


def test_int32_bounds_hold_an_out_of_range_value() -> None:
    """Check a value past an Int32 column's range fails the field, so Validate holds the record."""

    class Key(Record, table="cipher.k", writer=Upsert(("source_id", "date_key"))):
        record_key: ClassVar[tuple[str, ...]] = ("day",)
        record_version: ClassVar[int] = 1

        day: Annotated[Int32, To("date_key")]

    assert Key.model_validate({"day": 2**31 - 1}).day == 2**31 - 1
    with pytest.raises(ValueError, match="less than or equal to 2147483647"):
        Key.model_validate({"day": 2**31})


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("Annotated[str, To('n'), Width(32)]", "a Width mark is for an int field"),
        ("Annotated[Decimal, Field(max_digits=10, decimal_places=2), Width(32), To('n')]",
         "a Width mark is for an int field"),
        ("Annotated[int, Width(16), To('n')]", "Width\\(16\\) needs bounds within -32768 to 32767; use Int16"),
        ("Annotated[int, Ge(0), Width(16), To('n')]", "Width\\(16\\) needs bounds"),
        ("Annotated[Int32, Width(16), To('n')]", "a field takes one Width mark, got 2"),
        ("Annotated[int, Ge(-(2**31)), Le(2**31), Width(32), To('n')]", "Width\\(32\\) needs bounds"),
        ("Annotated[int, Gt(-(2**15) - 2), Lt(2**15), Width(16), To('n')]", "Width\\(16\\) needs bounds"),
    ],
)  # fmt: skip
def test_width_needs_one_mark_on_an_int_bounded_within_its_width(field: str, message: str) -> None:
    """Check a Width mark is refused off an int, beside another Width, or with bounds wider than the column's range."""
    code = (
        "class Bad(Record, table='cipher.bad', writer=Upsert(('source_id', 'code'))):\n"
        "    record_key: ClassVar[tuple[str, ...]] = ('id',)\n"
        "    record_version: ClassVar[int] = 1\n"
        "    id: Annotated[str, To('code')]\n"
        f"    n: {field}\n"
    )
    names = {
        "Annotated": Annotated, "ClassVar": ClassVar, "Decimal": Decimal, "Field": Field, "Ge": Ge, "Gt": Gt,
        "Int32": Int32, "Le": Le, "Lt": Lt, "Record": Record, "To": To, "Upsert": Upsert, "Width": Width,
    }  # fmt: skip
    with pytest.raises(TypeError, match=message):
        exec(code, names)  # noqa: S102


def test_width_accepts_bounds_narrower_than_its_width() -> None:
    """Check an int bounded inside the width's range, such as a non-negative Int16, loads into the narrow column."""

    class Coded(Record, table="cipher.coded", writer=Upsert(("source_id", "code"))):
        record_key: ClassVar[tuple[str, ...]] = ("id",)
        record_version: ClassVar[int] = 1

        id: Annotated[str, To("code")]
        status: Annotated[Int16, Ge(0), To("status")]
        hour: Annotated[int, Gt(-1), Lt(24), Width(16), To("hour")]

    found = {column.name: column.postgres for column in target_of(Coded).columns}
    assert found["status"] == "smallint"
    assert found["hour"] == "smallint"


def test_width_mark_refusals() -> None:
    """Check Width refuses a width other than 32 or 16, and a field that is not an int."""
    with pytest.raises(ValueError, match="Width needs 32 or 16 bits"):
        Width(8)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="a Width mark is for an int field"):

        class Bad(Record, table="cipher.bad", writer=Upsert(("source_id", "code"))):
            record_key: ClassVar[tuple[str, ...]] = ("id",)
            record_version: ClassVar[int] = 1

            id: Annotated[str, To("code"), Width(32)]


DAYS = {"valid_from": "valid_from_key", "valid_to": "valid_to_key"}


class Bundle(Part, table="cipher.link_sku_bundle", writer=ReplaceChildren(("component_sku_id",))):
    """A bundle component, replaced under its SKU."""

    component: Annotated[str, To("component_sku_id")]
    quantity: Annotated[int, To("quantity")]


class SkuPrice(Part, table="cipher.dim_sku_price", writer=Versioned((), **DAYS)):
    """A SKU's price: a one-to-one Versioned child keyed by the SKU it inherits."""

    price: Annotated[int, To("retail_price")]


class ProductSku(Part, table="cipher.dim_sku", writer=Versioned(("tiktok_sku_id",), **DAYS)):
    """A versioned SKU under its product, keyed by its own id."""

    id: Annotated[str, To("tiktok_sku_id")]
    price: Annotated[SkuPrice | None, Rows()] = None
    bundle: Annotated[list[Bundle], Rows()] = Field(default_factory=list)


class Product(
    Record,
    table="cipher.dim_product",
    writer=Versioned(("shop_id", "tiktok_product_id"), **DAYS),
    source_column="shop_id",
):
    """A versioned product with its SKUs."""

    record_key: ClassVar[tuple[str, ...]] = ("id",)
    record_version: ClassVar[int] = 1

    id: Annotated[str, To("tiktok_product_id")]
    skus: Annotated[list[ProductSku], Rows()] = Field(default_factory=list)


def test_children_nest_under_a_versioned_product_with_their_inherited_columns() -> None:
    """Check each kind of child gets its inherited columns first and its row key by rule, at every depth."""
    found = {
        target.table: ([c.name for c in target.columns], target.inherited, row_key(target))
        for target in walk(target_of(Product))
    }
    assert list(found) == ["cipher.dim_product", "cipher.dim_sku", "cipher.dim_sku_price", "cipher.link_sku_bundle"]
    assert found == {
        "cipher.dim_product": (["shop_id", "tiktok_product_id"], (), ("shop_id", "tiktok_product_id")),
        # a keyed child: the parent's key without the source column, as a plain column; keyed by its own key
        "cipher.dim_sku": (["tiktok_product_id", "tiktok_sku_id"], ("tiktok_product_id",), ("tiktok_sku_id",)),
        # a one-to-one child with no key of its own: keyed by what it inherits, the SKU's row key
        "cipher.dim_sku_price": (["tiktok_sku_id", "retail_price"], ("tiktok_sku_id",), ("tiktok_sku_id",)),
        # a ReplaceChildren child of a child: the SKU's row key, then its own key
        "cipher.link_sku_bundle": (
            ["tiktok_sku_id", "component_sku_id", "quantity"],
            ("tiktok_sku_id",),
            ("tiktok_sku_id", "component_sku_id"),
        ),
    }


def test_a_replace_children_child_inherits_its_parents_whole_row_key() -> None:
    """Check a ReplaceChildren child of a ReplaceChildren child inherits the child's row key, source column included."""

    class Note(Part, table="sales.note", writer=ReplaceChildren(("note_id",))):
        note_id: Annotated[str, To("note_id")]

    class Noted(Part, table="sales.noted", writer=ReplaceChildren(("sku_id",))):
        sku_id: Annotated[str, To("sku_id")]
        notes: Annotated[list[Note], Rows()]

    class Sale(Record, table="sales.sale", writer=Upsert(("source_id", "order_id"))):
        record_key: ClassVar[tuple[str, ...]] = ("order_id",)
        record_version: ClassVar[int] = 1

        order_id: Annotated[str, To("order_id")]
        lines: Annotated[list[Noted], Rows()]

    note = dict(dict(target_of(Sale).children)["lines"].children)["notes"]
    assert row_key(note) == ("source_id", "order_id", "sku_id", "note_id")


def test_an_upsert_child_may_be_one_to_one_with_no_key() -> None:
    """Check a single-part Upsert child with an empty key is keyed by its inherited columns, and None is allowed."""

    class Address(Part, table="sales.address", writer=Upsert(())):
        city: Annotated[str, To("city")]

    class Customer(Record, table="sales.customer", writer=Upsert(("source_id", "customer_id"))):
        record_key: ClassVar[tuple[str, ...]] = ("customer_id",)
        record_version: ClassVar[int] = 1

        customer_id: Annotated[str, To("customer_id")]
        address: Annotated[Address | None, Rows()] = None

    address = dict(target_of(Customer).children)["address"]
    assert [c.name for c in address.columns] == ["customer_id", "city"]
    assert row_key(address) == ("customer_id",)


def test_an_upsert_child_key_may_name_an_inherited_column() -> None:
    """Check an Upsert list child keyed by its parent's id and its own is keyed by both, the inherited one first."""

    class Line(Part, table="sales.request_line", writer=Upsert(("request_id", "line_id"))):
        line_id: Annotated[str, To("line_id")]
        refund: Annotated[Decimal, Field(max_digits=12, decimal_places=2), To("refund")]

    class Request(Record, table="sales.request", writer=Upsert(("source_id", "request_id"))):
        record_key: ClassVar[tuple[str, ...]] = ("request_id",)
        record_version: ClassVar[int] = 1

        request_id: Annotated[str, To("request_id")]
        lines: Annotated[list[Line], Rows()]

    line = dict(target_of(Request).children)["lines"]
    assert [c.name for c in line.columns] == ["request_id", "line_id", "refund"]
    assert line.inherited == ("request_id",)
    assert row_key(line) == ("request_id", "line_id")


def test_an_upsert_child_key_naming_a_column_it_does_not_inherit_is_refused() -> None:
    """Check an Upsert child's key column that is neither its own nor inherited is refused when the parent adopts it."""

    class Line(Part, table="sales.request_line", writer=Upsert(("order_id", "line_id"))):
        line_id: Annotated[str, To("line_id")]

    with pytest.raises(ValueError, match="names 'order_id', which the child neither has nor inherits"):

        class Request(Record, table="sales.request", writer=Upsert(("source_id", "request_id"))):
            record_key: ClassVar[tuple[str, ...]] = ("request_id",)
            record_version: ClassVar[int] = 1

            request_id: Annotated[str, To("request_id")]
            lines: Annotated[list[Line], Rows()]


def test_a_record_key_naming_a_column_it_does_not_have_is_refused() -> None:
    """Check only a Part's Upsert key may name a column it does not have: a Record's is refused at once."""
    with pytest.raises(ValueError, match="names 'order_id', which is not a target column"):

        class Request(Record, table="sales.request", writer=Upsert(("source_id", "order_id", "request_id"))):
            record_key: ClassVar[tuple[str, ...]] = ("request_id",)
            record_version: ClassVar[int] = 1

            request_id: Annotated[str, To("request_id")]


@pytest.mark.parametrize(
    ("child", "field", "message"),
    [
        ("class C(Part, table='t.c', writer=Snapshot(('k',))):\n    k: Annotated[str, To('k')]",
         "Annotated[list[C], Rows()]", "writer must be Upsert, Versioned or ReplaceChildren"),
        ("class C(Part, table='t.c', writer=Update(('k',))):\n    k: Annotated[str, To('k')]\n"
         "    n: Annotated[int, To('n')]", "Annotated[list[C], Rows()]", "writer must be Upsert, Versioned or"),
        ("class C(Part, table='t.c', writer=ReplacePartition('d')):\n    d: Annotated[date, To('d')]",
         "Annotated[C, Rows()]", "writer must be Upsert, Versioned or ReplaceChildren"),
        ("class C(Part, table='t.c', writer=Upsert(('source_id', 'k')), source_column='source_id'):\n"
         "    k: Annotated[str, To('k')]", "Annotated[list[C], Rows()]", "a Rows child has no source_column"),
        ("class C(Part, table='t.c', writer=Upsert(())):\n    k: Annotated[str, To('k')]",
         "Annotated[list[C], Rows()]", "a list's items need a key of their own"),
        ("class C(Part, table='t.c', writer=Versioned((), valid_from='f', valid_to='t')):\n"
         "    k: Annotated[str, To('k')]", "Annotated[list[C], Rows()]", "a list's items need a key of their own"),
        ("class C(Part, table='t.c', writer=Versioned(('k',), valid_from='f', valid_to='t', prune=False)):\n"
         "    k: Annotated[str, To('k')]", "Annotated[list[C], Rows()]", "so it cannot set prune=False"),
        ("class C(Part, table='t.c', writer=Versioned(('k',), valid_from='f', valid_to='t', allow_empty=True)):\n"
         "    k: Annotated[str, To('k')]", "Annotated[list[C], Rows()]", "so it does not set allow_empty"),
        ("class C(Part):\n    k: Annotated[str, To('k')]", "Annotated[C, Rows()]",
         "Rows needs SomePart or SomePart \\| None, where SomePart declares a table"),
        ("class C(Part, table='t.c', writer=Upsert(('k',))):\n    k: Annotated[str, To('k')]",
         "Annotated[list[C] | None, Rows()] = None", "Rows needs list\\[SomePart\\]"),
        ("class C(Part, table='t.c', writer=Upsert(('k',))):\n    k: Annotated[str, To('k')]",
         "Annotated[int, Rows()] = 0", "Rows is only for a part or a list of parts"),
        ("class C(Part, table='t.c', writer=Upsert(('order_id',))):\n    k: Annotated[str, To('order_id')]",
         "Annotated[C, Rows()]", "columns named more than once: order_id"),
        ("class C(Part, table='t.order', writer=Upsert(('k',))):\n    k: Annotated[str, To('k')]",
         "Annotated[list[C], Rows()]", "more than one target loads into t.order"),
    ],
    ids=[
        "snapshot child", "update child", "replace partition child", "child with a source column",
        "list of keyless upserts", "list of keyless versions", "child with prune off", "child allowing empty",
        "part without a table",
        "nullable list",
        "rows on a scalar", "inherited name clashes", "table clash",
    ],
)  # fmt: skip
def test_rows_child_refused(child: str, field: str, message: str) -> None:
    """Check a Rows child must be a keyed or ReplaceChildren part with no source column, fitting its parent."""
    code = (
        f"{child}\n"
        "class Parent(Record, table='t.order', writer=Upsert(('source_id', 'order_id'))):\n"
        "    record_key: ClassVar[tuple[str, ...]] = ('order_id',)\n"
        "    record_version: ClassVar[int] = 1\n"
        "    order_id: Annotated[str, To('order_id')]\n"
        f"    child: {field}\n"
    )
    names = {
        "Annotated": Annotated, "ClassVar": ClassVar, "date": date, "Part": Part, "Record": Record,
        "ReplaceChildren": ReplaceChildren, "ReplacePartition": ReplacePartition, "Rows": Rows, "Snapshot": Snapshot,
        "To": To, "Update": Update, "Upsert": Upsert, "Versioned": Versioned,
    }  # fmt: skip
    with pytest.raises(TypeError, match=message):
        exec(code, names)  # noqa: S102


def test_a_table_clash_deep_in_the_tree_is_refused() -> None:
    """Check a grandchild that loads into its grandparent's table is refused when the grandparent is built."""

    class Deep(Part, table="t.top", writer=ReplaceChildren(("d",))):
        d: Annotated[str, To("d")]

    class Middle(Part, table="t.middle", writer=Upsert(("m",))):
        m: Annotated[str, To("m")]
        deep: Annotated[list[Deep], Rows()]

    with pytest.raises(TypeError, match=r"more than one target loads into t\.top"):

        class Top(Record, table="t.top", writer=Upsert(("source_id", "k"))):
            record_key: ClassVar[tuple[str, ...]] = ("k",)
            record_version: ClassVar[int] = 1

            k: Annotated[str, To("k")]
            middles: Annotated[list[Middle], Rows()]


def test_rows_refused_under_update_in_a_nested_part_and_under_a_keyless_child() -> None:
    """Check Rows stays refused under an Update table, inside a nested part, and under a one-to-one keyless child."""
    with pytest.raises(TypeError, match="Rows is not for an Update table"):

        class Patch(Record, table="t.patch", writer=Update(("source_id", "k"))):
            record_key: ClassVar[tuple[str, ...]] = ("k",)
            record_version: ClassVar[int] = 1

            k: Annotated[str, To("k")]
            n: Annotated[int, To("n")]
            price: Annotated[SkuPrice | None, Rows()] = None

    with pytest.raises(TypeError, match="Rows is only for a field of the class that declares the table"):

        class Inner(Part):
            price: Annotated[SkuPrice | None, Rows()] = None

        class Outer(Record, table="t.outer", writer=Upsert(("source_id", "k"))):
            record_key: ClassVar[tuple[str, ...]] = ("k",)
            record_version: ClassVar[int] = 1

            k: Annotated[str, To("k")]
            inner: Inner

    with pytest.raises(TypeError, match="Rows needs a writer with a key of its own"):

        class Keyless(Part, table="t.keyless", writer=Upsert(())):
            n: Annotated[int, To("n")]
            lines: Annotated[list[Line], Rows()]


def test_a_part_with_a_keyed_writer_is_not_a_top_level_target() -> None:
    """Check target_of refuses a Part, whatever its writer, since it loads only as a child."""
    with pytest.raises(TypeError, match="ProductSku loads as child rows"):
        target_of(ProductSku)


def test_an_empty_key_is_refused_outside_a_one_to_one_child() -> None:
    """Check only Upsert and Versioned allow an empty key, and a Record never does."""
    with pytest.raises(ValueError, match="the writer's key must name distinct columns"):

        class Gone(Part, table="t.gone", writer=Snapshot(())):
            n: Annotated[int, To("n")]

    with pytest.raises(TypeError, match="the writer's key must be the columns k"):

        class Root(Record, table="t.root", writer=Upsert(()), source_column=None):
            record_key: ClassVar[tuple[str, ...]] = ("k",)
            record_version: ClassVar[int] = 1

            k: Annotated[str, To("k")]


def test_a_one_to_one_versioned_child_needs_a_unique_key_on_its_inherited_columns() -> None:
    """Check mismatches wants a one-to-one Versioned child's unique key on its inherited columns and valid_from."""
    price = dict(dict(target_of(Product).children)["skus"].children)["price"]

    def col(name: str, kind: str, nullable: bool) -> TableColumn:
        return TableColumn(name=name, type=kind, nullable=nullable, has_default=False, writable=True)

    columns = {
        "tiktok_sku_id": col("tiktok_sku_id", "text", False),
        "retail_price": col("retail_price", "bigint", False),
        "valid_from_key": col("valid_from_key", "integer", False),
        "valid_to_key": col("valid_to_key", "integer", True),
    }
    fits = TableInfo(
        table="cipher.dim_sku_price",
        columns=columns,
        unique_keys=frozenset({frozenset({"tiktok_sku_id", "valid_from_key"})}),
    )
    assert mismatches(price, fits) == []
    narrow = TableInfo(
        table="cipher.dim_sku_price", columns=columns, unique_keys=frozenset({frozenset({"valid_from_key"})})
    )
    assert mismatches(price, narrow) == [
        "cipher.dim_sku_price has no unique key on exactly tiktok_sku_id, valid_from_key"
    ]


class DaySales(Part, table="cipher.fact_product_sales", writer=ReplaceChildren(("source_key",))):
    """A product-day's sales by traffic source, replaced under its product-day."""

    source_key: Annotated[int, To("source_key")]
    gmv: Annotated[int, To("gmv")]


class DayRating(Part, table="cipher.fact_product_ratings", writer=Upsert(())):
    """A product-day's rating: one-to-one, keyed by the product-day it inherits."""

    stars: Annotated[int, To("stars")]


class ProductDay(Record, container=True):
    """One product's day of performance: no table of its own, only its facts."""

    record_key: ClassVar[tuple[str, ...]] = ("product", "day")
    record_version: ClassVar[int] = 1

    product: Annotated[str, To("tiktok_product_id")]
    day: Annotated[Int32, To("date_key")]
    sales: Annotated[list[DaySales], Rows()] = Field(default_factory=list)
    rating: Annotated[DayRating | None, Rows()] = None


def test_a_container_has_no_table_and_its_children_inherit_its_key_and_source() -> None:
    """Check a container's target has no table or writer, and every child inherits its source and key columns."""
    found = {
        target.table: ([c.name for c in target.columns], target.inherited, row_key(target))
        for target in walk(target_of(ProductDay))
    }
    key = ("source_id", "tiktok_product_id", "date_key")
    assert found == {
        None: (list(key), (), key),
        "cipher.fact_product_sales": ([*key, "source_key", "gmv"], key, (*key, "source_key")),
        "cipher.fact_product_ratings": ([*key, "stars"], key, key),
    }
    assert target_of(ProductDay).writer is None
    assert mismatches(target_of(ProductDay), None) == []  # nothing to compare


def test_a_container_without_a_source_column() -> None:
    """Check a container may opt out of the source column, so its children are not per source."""

    class ShopDay(Record, container=True, source_column=None):
        record_key: ClassVar[tuple[str, ...]] = ("day",)
        record_version: ClassVar[int] = 1

        day: Annotated[Int32, To("date_key")]
        rating: Annotated[DayRating | None, Rows()] = None

    rating = dict(target_of(ShopDay).children)["rating"]
    assert rating.inherited == ("date_key",)
    assert row_key(rating) == ("date_key",)


def test_a_subclass_of_a_container_is_not_one() -> None:
    """Check a subclass that does not repeat container=True declares no target, as with a table."""

    class Copy(ProductDay): ...

    assert Copy.__target__ is None


@pytest.mark.parametrize(
    ("keywords", "fields", "message"),
    [
        ("container=True, table='t.x', writer=Upsert(('source_id', 'k'))", "sales: Annotated[list[DaySales], Rows()]",
         "a container has no table or writer"),
        ("container=True", "sales: Annotated[list[DaySales], Rows()]\n    extra: Annotated[int, To('extra')] = 0",
         "only record_key fields take a To: extra"),
        ("container=True", "", "a container loads only its Rows children, so it needs one"),
        ("container=True", "sales: Annotated[list[Gone], Rows()]", "writer must be Upsert, Versioned or"),
        ("container=True", "sales: Annotated[list[DaySales], Rows()]\n"
         "    more: Annotated[list[DaySales], Rows()] = Field(default_factory=list)",
         "more than one target loads into cipher.fact_product_sales"),
    ],
    ids=["table and writer", "non-key To", "no Rows", "snapshot child", "table clash"],
)  # fmt: skip
def test_container_refused(keywords: str, fields: str, message: str) -> None:
    """Check a container takes no table or writer, loads only key fields, needs Rows, and keeps the child rules."""
    code = (
        f"class Refused(Record, {keywords}):\n"
        "    record_key: ClassVar[tuple[str, ...]] = ('k',)\n"
        "    record_version: ClassVar[int] = 1\n"
        "    k: Annotated[str, To('k')]\n"
        f"    {fields}\n"
    )

    class Gone(Part, table="cipher.gone", writer=Snapshot(("n",))):
        n: Annotated[int, To("n")]

    names = {
        "Annotated": Annotated, "ClassVar": ClassVar, "DaySales": DaySales, "Field": Field, "Gone": Gone,
        "Record": Record, "Rows": Rows, "To": To, "Upsert": Upsert,
    }  # fmt: skip
    with pytest.raises(TypeError, match=message):
        exec(code, names)  # noqa: S102


def test_a_container_key_field_needs_a_to_and_a_part_cannot_be_one() -> None:
    """Check a container's record_key field needs a To, and only a Record can be a container."""
    with pytest.raises(TypeError, match=r"^Untold\.k is in record_key, so it needs a To$"):

        class Untold(Record, container=True):
            record_key: ClassVar[tuple[str, ...]] = ("k",)
            record_version: ClassVar[int] = 1

            k: str
            rating: Annotated[DayRating | None, Rows()] = None

    with pytest.raises(TypeError, match="only a Record can be a container"):

        class Holder(Part, container=True):
            rating: Annotated[DayRating | None, Rows()] = None


def test_rows_is_refused_when_the_child_would_inherit_no_column() -> None:
    """Check Rows needs the parent's key to pass its child a column besides the source column."""

    class Note(Part, table="t.note", writer=Upsert(("k",))):
        k: Annotated[str, To("k")]

    with pytest.raises(TypeError, match=r"Shop\.notes: Rows needs its parent's key to hold a column besides"):

        class Shop(Part, table="t.shop", writer=Upsert(("source_id",)), source_column="source_id"):
            name: Annotated[str, To("name")]
            notes: Annotated[list[Note], Rows()]
