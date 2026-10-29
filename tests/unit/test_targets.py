"""Tests for declaring a model's target table with class keywords."""

from datetime import date
from typing import Annotated, ClassVar

import pytest

from pipeline.checks.record import Part, Record
from pipeline.checks.targets import (
    ReplaceChildren,
    ReplacePartition,
    Rows,
    Snapshot,
    To,
    Upsert,
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
