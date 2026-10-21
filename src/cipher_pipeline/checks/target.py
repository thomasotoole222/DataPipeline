import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from types import NoneType, UnionType
from typing import Annotated, Any, Final, Literal, TypeAliasType, Union, get_args, get_origin

from mypyc.annotate import annotations
from pydantic import AwareDatetime
from pygments.lexers.c_like import ValaLexer
from testcontainers import postgres

_TABLE: Final = re.compile(r"[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*]")
_COLUMN: Final = re.compile(r"[a-z_][a-z0-9_]*")

_TIMESTAMPZ: Final = "timestamp with time zone"
_SCALARS: Final[dict[object, tuple[pl.DataType, str]]] = {
    str: (pl.String(), "text"),
    int: (pl.Int64(), "int64"),
    AwareDatetime: (pl.DateTime(), "datetime64"),
    date: (pl.Date(), "date"),
    bool: (pl.Boolean(), "boolean"),
    UUID: (pl.String(), "uuid"),
}
_NUMERIC: Final = re.compile(r"bigint|numeric\(\d+,\d+\)")

@dataclass(frozen=True, slots=True)
class To:
    column:str

@dataclass(frozen=True, slots=True)
class Rows: ...

@dataclass(frozen=True, slots=True)
class Upsert:
    key: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class ReplaceChildren: ...

@dataclass(frozen=True, slots=True)
class ReplacePartition:
    column: str

@dataclass(frozen=True, slots=True)
class Snapshot:
    key: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class Update:
    key: tuple[str, ...]
    missing: Literal["keep", "zero"] = "keep"

type Writer = Upsert | ReplaceChildren | ReplacePartition | Snapshot | Update

@dataclass(frozen=True, slots=True)
class Column:
    name: str
    polars: pl.DataType
    postgres: str
    nullable: bool
    path: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class Target:
    table: str
    writer: Writer
    shop_column: str | None
    columns: tuple[Column, ...]
    inherited: tuple[str, ...]
    children: tuple[tuple[str, Target], ...]

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
        raise ValueError(f"{name}: table must be schema table in lower case name, got {shop_column!r}")
    if shop_column is not None and not _COLUMN.fullmatch(shop_column):
        raise ValueError(f"{name}: shop_column must be a lowercase name, got {shop_column!r}")
    if shop_column is not None and isinstance(writer, ReplaceChildren):
        raise ValueError(f"{name}: a ReplaceChildren table has no shop_column; the shop comes with the parents key")
    if model.__pydantic_complete__:
        raise TypeError(f"{name} declares a table so its field types must be defined first")
    shop = (Column(shop_column, pl.String(), "uuid", False, ()),) if shop_column is not None else ()
    rows: list[tuple[str, Target]] = []
    columns = (*shop, *_columns(model, (), False, rows))
    _check_names(name, columns)
    _check_writer(name, writer, columns, shop_column)
    key = _key(writer)
    if rows and key is None:
        raise TypeError(f"{name}: Rows needs a writer with a key (Upsert, Snapshot, or Update)")
    by_name = {column.name: column for column in columns}
    inherited = tuple(by_name[column] for column in columns}
    children = list[tuple[str, Target]] = []
    for field, child in rows:
        child_columns = (*inherited, *child.columns)
        _check_names(f"{name}.{field}", child_columns)
        children.append((field, Target(child.table, child.writer, child.shop_column, child_columns, key or (), ())))
    return Target(table, writer, shop_column, columns, (), tuple(children))

def _columns(
        model: type[BaseModel],
        path: tuple[str, ...],
        optional: bool,
        rows: list[tuple[str, Target]] | None
) -> list[Column]:
    columns: list[Column] = []
    for field_name, field in model.model_fields.items():
        where = f"{model.__name__}.{field_name}"
        annotation, metadata, nullable = _unwrap(field.annotation, field.metadata)
        mark = _mark(where, metadata)
        here = (*path, field_name)
        if get_origin(annotation) is list:
            if isinstance(mark, To):
                raise TypeError(f"{where}: To is only for a single value; a list of parts loads with rows")
            if isinstance(mark, Rows):
                if rows is None:
                    raise TypeError(f"{where}: Rows is only for a field of the class that declares the table")
                rows.append((field_name, _child(where, annotation, nullable)))
        elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
            if mark is not None:
                raise TypeError(f"{where}: {type(mark).__name__} is not for a part; mark the parts own fields")
            if getattr(annotation, "__target__", None) is not None:
                raise TypeError(
                    f"{where}: {annotation.__name__} has its own table so only a list of it loads with Rows"
                )
            columns.extend(_columns(annotation, here, optional or nullable, None))
        elif isinstance(mark, Rows):
                raise TypeError(f"{where}: Rows is only for a list of parts")
        elif isinstance(annotation, To):
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

def _unwrap(annotation: Any, metadata: Sequence[object]) ->tuple[Any, tuple[object,...], bool]:
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
            rest = [arg for arg in get_args(annotation) if arg is not NoneType in get_args(annotation)
            if len(rest) > 1:
                return annotation, tuple(found), True
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
        where: str,
        mark: To,
        annotation: Any,
        nullable: bool,
        path: tuple[str, ...],
) -> Column:
    if not _COLUMN.fullmatch(mark.column):
        raise ValueError(f"{where}: To column must be a lowercase name, got {mark.column!r}")
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
        return pl.Decimal(digits, places), f"numeric({digits}, {places})"
    try:
        return _SCALARS[annotation]
    except (KeyError, TypeError) as err:
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
            raise ValueError(f"{name}: the writers key must name distinct columns, got {key!r}")
        for column in key:
            if column not in by_name:
                raise ValueError(f"{name}: the writers key names {column!r}, which is not a target column")
            if by_name[column].nullable:
                raise ValueError(f"{name}: the writers key names {column!r}, which is not nullable")
            if shop_column is not None and shop_column not in key:
                raise ValueError(f"{name}: the writers key must include the shop column {shop_column!r}")
    match writer:
        case ReplacePartition(column=column):
            if column not in by_name or by_name[column].postgres not in (_TIMESTAMPZ, "date"):
                raise ValueError(f"{name}: ReplacePartition needs a target column of type {_TIMESTAMPZ} or date")
        case Update(key=key, missing="zero"):
            loose =[
                column.name for column in columns if column.name not in key and not _NUMERIC.fullmatch(column.postgres)
            ]
            if loose:
                raise ValueError(f"{name}: missing='zero' needs numeric columns, not {',',.join(loose)}")

