import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from types import NoneType, UnionType
from typing import Annotated, Any, Final, Literal, TypeAliasType, Union, get_args, get_origin

from pydantic import AwareDatetime

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