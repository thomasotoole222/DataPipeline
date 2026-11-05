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
    Upsert,
    Versioned,
    Width,
    Writer,
    mismatches,
    row_key,
    target_of,
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
    ],
)  # fmt: skip
def test_versioned_declarations_refused(writer: str, field: str, message: str) -> None:
    """Check a Versioned writer refuses equal day columns, a field loading into one, a bad opens_from, no source."""
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


def test_versioned_refuses_rows_children() -> None:
    """Check a Versioned table cannot own Rows children: its rows are versions, not one row per parent."""
    with pytest.raises(TypeError, match="Rows is not for a Versioned table"):

        class Kit(Record, table="cipher.kit", writer=Versioned(("source_id", "kit"), valid_from="f", valid_to="t")):
            record_key: ClassVar[tuple[str, ...]] = ("id",)
            record_version: ClassVar[int] = 1

            id: Annotated[str, To("kit")]
            lines: Annotated[list[Line], Rows()]


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
