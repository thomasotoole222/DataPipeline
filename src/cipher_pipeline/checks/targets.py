import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from types import NoneType, UnionType
from typing import Annotated, Any, Final, Literal, TypeAliasType, Union, get_args, get_origin
from uuid import UUID

import polars as pl
from pydantic import AwareDatetime, BaseModel
from pydantic.fields import FieldInfo

from cipher_pipeline.db.catalog import TableInfo

_TABLE: Final = re.compile(r"[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*")
_COLUMN: Final = re.compile(r"[a-z_][a-z0-9_]*")

_TIMESTAMPTZ: Final = "timestamp with time zone"
_SCALARS: Final[dict[object, tuple[pl.DataType, str]]] = {
    str: (pl.String(), "text"),
    int: (pl.Int64(), "bigint"),
    AwareDatetime: (pl.Datetime("us", "UTC"), _TIMESTAMPTZ),  # "us": microseconds, the precision timestamptz stores
    date: (pl.Date(), "date"),
    bool: (pl.Boolean(), "boolean"),
    UUID: (pl.String(), "uuid"),
}
_NUMERIC: Final = re.compile(r"bigint|numeric\(\d+,\d+\)")


@dataclass(frozen=True, slots=True)
class To:
    column: str


@dataclass(frozen=True, slots=True)
class Rows: ...


@dataclass(frozen=True, slots=True)
class Upsert:
    key: tuple[str, ...]  # target columns, the shop column included when the table has one


@dataclass(frozen=True, slots=True)
class ReplaceChildren: ...  # key = the columns inherited from the parent


@dataclass(frozen=True, slots=True)
class ReplacePartition:
    column: str  # a timestamptz or date column; rows with it in the task's frame are replaced


@dataclass(frozen=True, slots=True)
class Snapshot:
    key: tuple[str, ...]  # target columns, the shop column included when the table has one


@dataclass(frozen=True, slots=True)
class Update:
    key: tuple[str, ...]  # target columns, the shop column included when the table has one
    missing: Literal["keep", "zero"] = "keep"  # what happens to the shop's rows the source did not return


type Writer = Upsert | ReplaceChildren | ReplacePartition | Snapshot | Update


@dataclass(frozen=True, slots=True)
class Column:
    name: str  # target column
    polars: pl.DataType  # frame type
    postgres: str  # exactly as Postgres format_type() prints it
    nullable: bool
    path: tuple[
        str, ...
    ]  # where the value is on the record: field names, nested parts included; () for the shop column


@dataclass(frozen=True, slots=True)
class Target:
    table: str  # "schema.table"
    writer: Writer
    shop_column: str | None
    columns: tuple[
        Column, ...
    ]  # every column the frame has, in frame order: shop column first, then inherited, then own
    inherited: tuple[str, ...]  # columns copied from the parent row; () for a top-level target
    children: tuple[tuple[str, Target], ...]  # (the Rows field name, the child's target), in field order


def target_of(model: type[BaseModel]) -> Target:
    target: Target | None = getattr(model, "__target__", None)
    if target is None:
        raise TypeError(f"{model.__name__} declares no table")
    if isinstance(target.writer, ReplaceChildren):
        raise TypeError(f"{model.__name__} loads as child rows; its target is in its parents children")
    return target


def build_target(model: type[BaseModel], table: str, writer: Writer, shop_column: str | None) -> Target:
    name = model.__name__
    if not _TABLE.fullmatch(table):
        raise ValueError(f"{name}: table must be schema.table in lower case, got {table!r}")
    if shop_column is not None and not _COLUMN.fullmatch(shop_column):
        raise ValueError(f"{name}: shop_column must be a lower-case name, got {shop_column!r}")
    if shop_column is not None and isinstance(writer, ReplaceChildren):
        raise TypeError(f"{name}: a ReplaceChildren table has no shop_column; the shop comes with the parent's key")
    if not model.__pydantic_complete__:
        raise TypeError(f"{name} declares a table, so its field types must be defined first")
    shop = (Column(shop_column, pl.String(), "uuid", False, ()),) if shop_column is not None else ()
    rows: list[tuple[str, Target]] = []
    columns = (*shop, *_columns(model, (), False, rows))
    _check_names(name, columns)
    _check_writer(name, writer, columns, shop_column)
    key = _key(writer)
    if rows and key is None:
        raise TypeError(f"{name}: Rows needs a writer with a key (Upsert, Snapshot or Update)")
    by_name = {column.name: column for column in columns}
    inherited = tuple(by_name[column] for column in key or ())  # Rows implies a key, checked above
    tables = [table, *(child.table for _, child in rows)]
    shared = sorted({t for t in tables if tables.count(t) > 1})
    if shared:
        raise TypeError(f"{name}: more than one target loads into {', '.join(shared)}")
    children: list[tuple[str, Target]] = []
    for field, child in rows:
        child_columns = (*inherited, *child.columns)
        _check_names(f"{name}.{field}", child_columns)
        children.append((field, Target(child.table, child.writer, child.shop_column, child_columns, key or (), ())))
    return Target(table, writer, shop_column, columns, (), tuple(children))


def _columns(
    model: type[BaseModel], path: tuple[str, ...], optional: bool, rows: list[tuple[str, Target]] | None
) -> list[Column]:
    columns: list[Column] = []
    for field_name, field in model.model_fields.items():
        where = f"{model.__name__}.{field_name}"
        annotation, metadata, nullable = _unwrap(field.annotation, field.metadata)
        mark = _mark(where, metadata)
        here = (*path, field_name)
        if get_origin(annotation) is list:
            if isinstance(mark, To):
                raise TypeError(f"{where}: To is only for a single value; a list of parts loads with Rows")
            if isinstance(mark, Rows):
                if rows is None:
                    raise TypeError(f"{where}: Rows is only for a field of the class that declares the table")
                rows.append((field_name, _child(where, annotation, nullable)))
        elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
            if mark is not None:
                raise TypeError(f"{where}: {type(mark).__name__} is not for a part; mark the part's own fields")
            if getattr(annotation, "__target__", None) is not None:
                raise TypeError(
                    f"{where}: {annotation.__name__} has its own table, so only a list of it loads, with Rows"
                )
            columns.extend(_columns(annotation, here, optional or nullable, None))
        elif isinstance(mark, Rows):
            raise TypeError(f"{where}: Rows is only for a list of parts")
        elif isinstance(mark, To):
            columns.append(_column(where, mark, annotation, metadata, optional or nullable, here))
    for field_name, computed in model.model_computed_fields.items():
        where = f"{model.__name__}.{field_name}"
        annotation, metadata, nullable = _unwrap(computed.return_type, ())
        mark = _mark(where, metadata)
        if isinstance(mark, Rows):
            raise TypeError(f"{where}: Rows is not for a computed field")
        if isinstance(mark, To):
            columns.append(_column(where, mark, annotation, metadata, optional or nullable, (*path, field_name)))
    return columns


def _unwrap(annotation: Any, metadata: Sequence[object]) -> tuple[Any, tuple[object, ...], bool]:
    found = list(metadata)
    nullable = False
    while True:
        if isinstance(annotation, TypeAliasType):
            annotation = annotation.__value__
        elif get_origin(annotation) is Annotated:
            annotation, *extra = get_args(annotation)
            for item in extra:
                found.extend(item.metadata if isinstance(item, FieldInfo) else [item])
        elif get_origin(annotation) in (Union, UnionType) and NoneType in get_args(annotation):
            rest = [arg for arg in get_args(annotation) if arg is not NoneType]
            if len(rest) > 1:
                return annotation, tuple(found), True  # a union of several types; refused if it is loaded
            annotation, nullable = rest[0], True
        else:
            return annotation, tuple(found), nullable


def _mark(where: str, metadata: Sequence[object]) -> To | Rows | None:
    marks = [item for item in metadata if isinstance(item, To | Rows)]
    if len(marks) > 1:
        raise TypeError(f"{where} has more than one To or Rows")
    return marks[0] if marks else None


def _child(where: str, annotation: Any, nullable: bool) -> Target:
    (item,) = get_args(annotation)
    target: Target | None = getattr(item, "__target__", None)
    if nullable or target is None or not isinstance(target.writer, ReplaceChildren):
        raise TypeError(f"{where}: Rows needs list[SomePart], where SomePart declares a table with ReplaceChildren()")
    return target


def _column(
    where: str, mark: To, annotation: Any, metadata: Sequence[object], nullable: bool, path: tuple[str, ...]
) -> Column:
    if not _COLUMN.fullmatch(mark.column):
        raise ValueError(f"{where}: To column must be a lower-case name, got {mark.column!r}")
    polars, postgres = _column_type(where, annotation, metadata)
    return Column(mark.column, polars, postgres, nullable, path)


def _column_type(where: str, annotation: Any, metadata: Sequence[object]) -> tuple[pl.DataType, str]:
    if annotation is Decimal:
        found = {
            name: getattr(item, name)
            for item in metadata
            for name in ("max_digits", "decimal_places")
            if getattr(item, name, None) is not None
        }
        if len(found) < 2:
            raise TypeError(f"{where}: a loaded Decimal needs max_digits and decimal_places")
        digits, places = found["max_digits"], found["decimal_places"]
        return pl.Decimal(digits, places), f"numeric({digits},{places})"
    try:
        return _SCALARS[annotation]
    except (KeyError, TypeError) as err:  # TypeError: an unhashable annotation
        raise TypeError(f"{where}: {annotation!r} has no column type; see the type mapping") from err


def _check_names(where: str, columns: Sequence[Column]) -> None:
    names = [column.name for column in columns]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise TypeError(f"{where}: columns named more than once: {','.join(repeated)}")


def _check_writer(name: str, writer: Writer, columns: Sequence[Column], shop_column: str | None) -> None:
    by_name = {column.name: column for column in columns}
    key = _key(writer)
    if key is not None:
        if not key or len(set(key)) != len(key):
            raise ValueError(f"{name}: the writer's key must name distinct columns, got {key!r}")
        for column in key:
            if column not in by_name:
                raise ValueError(f"{name}: the writer's key names {column!r}, which is not a target column")
            if by_name[column].nullable:
                raise ValueError(f"{name}: the writer's key column {column!r} cannot allow None")
        if shop_column is not None and shop_column not in key:
            raise ValueError(f"{name}: the writer's key must include the shop column {shop_column!r}")
    match writer:
        case ReplacePartition(column=column):
            if column not in by_name or by_name[column].postgres not in (_TIMESTAMPTZ, "date"):
                raise ValueError(f"{name}: ReplacePartition needs a target column of type {_TIMESTAMPTZ} or date")
        case Snapshot():
            if "removed_at" in by_name:
                raise ValueError(f"{name}: a Snapshot sets removed_at itself, so no field may load into it")
        case Update(key=key, missing=missing):
            owned = [column for column in columns if column.name not in key]
            if not owned:
                raise ValueError(f"{name}: an Update needs a column outside its key to update")
            loose = [column.name for column in owned if not _NUMERIC.fullmatch(column.postgres)]
            if missing == "zero" and loose:
                raise ValueError(f"{name}: missing='zero' needs numeric columns, not {', '.join(loose)}")


def _key(writer: Writer) -> tuple[str, ...] | None:
    match writer:
        case Upsert(key=key) | Snapshot(key=key) | Update(key=key):
            return key
    return None


def mismatches(target: Target, info: TableInfo | None) -> list[str]:
    table = target.table
    if info is None:
        return [f"{table} does not exist or has 0 columns"]
    found: list[str] = []
    names = {column.name for column in target.columns}
    for column in target.columns:
        actual = info.columns.get(column.name)
        if actual is None:
            found.append(f"{table}.{column.name} does not exist")
        elif actual.type != column.postgres:
            found.append(f"{table}.{column.name} is {actual.type}, the model needs {column.postgres}")
        elif column.nullable and not actual.nullable:
            found.append(f"{table}.{column.name} is NOT NULL, but the model allows None")
        elif not actual.writable:
            found.append(f"{table}.{column.name} is generated, so the model cannot fill it")
    snapshot = isinstance(target.writer, Snapshot)
    for actual in info.columns.values():
        unfilled = not actual.nullable and not actual.has_default and actual.name not in names
        if unfilled and not (snapshot and actual.name == "removed_at"):
            found.append(f"{table}.{actual.name} is NOT NULL with no default, but the model does not fill it")
    key = _key(target.writer)
    if key is not None and frozenset(key) not in info.unique_keys:
        found.append(f"{table} has no unique key on ({', '.join(key)})")
    if snapshot:
        removed = info.columns.get("removed_at")
        if removed is None or removed.type != _TIMESTAMPTZ or not removed.nullable:
            found.append(f"{table}.removed_at must be a nullable {_TIMESTAMPTZ} column")
    return found
