"""Describe where a record model loads: its table, writer and columns, built from its field marks and type mapping.

Public API:
    TABLE_PATTERN: The pattern a ``schema.table`` name matches, used with ``fullmatch``.
    COLUMN_PATTERN: The pattern a bare column name matches, used with ``fullmatch``.
    To: Field mark that loads a single value into the named column.
    Rows: Field mark that loads a child part, or a ``list`` of them, as rows of the child's own table.
    Width: Field mark that loads an ``int`` field into a narrower integer column: ``integer`` or ``smallint``.
    Upsert: Writer that inserts rows or updates them on a conflict over ``key``.
    ReplaceChildren: Writer for a child table whose rows are replaced under each parent's key, matched on ``key``.
    ReplacePartition: Writer that replaces the rows whose ``column`` lies in the task's time frame.
    Snapshot: Writer keyed by ``key`` that sets ``removed_at`` itself.
    Update: Writer that updates existing rows matched on ``key``; ``missing`` says what happens to rows not returned.
    Versioned: A ``Snapshot`` that keeps history: a change closes the open version and opens a new one.
    Writer: The union of the writer classes.
    Column: One target column: its name, frame and Postgres types, nullability and path on the record.
    Target: A table, its writer, source column, columns and child targets; or a container's key columns and children.
    target_of: Return a model's top-level target.
    walk: Yield a target and every target under it, parents first.
    build_target: Build and check a model's target from its table, writer, source column and field marks.
    build_container: Build and check a container record's target: no table, only its key columns and ``Rows`` children.
    row_key: Return the columns that identify one row of a target's table, or None for a writer without them.
    mismatches: List the ways a table in the catalog does not fit a target.
    alias_value: Return a ``type`` alias's value, with a subscripted generic alias's arguments substituted.

Private units:
    _columns: Collect the columns of a model's marked fields and computed fields, nested parts included.
    _unwrap: Strip type aliases, ``Annotated`` and ``Optional`` from an annotation.
    _mark: Return the one ``To`` or ``Rows`` in a field's metadata, or None.
    _child: Return the target of a ``Rows`` field's child part, or of its list's item part.
    _adopt: Rebuild a child target, and every target under it, with the columns it inherits from its parent.
    _check_tables: Check that no two targets in a tree load into one table.
    _column: Build the ``Column`` for a ``To`` field.
    _column_type: Map an annotation to its frame type and Postgres type.
    _check_names: Check that no column name appears more than once.
    _check_writer: Check that a writer fits the columns.
    _key: Return a writer's key, or None for a writer without one.
    _day_kind: Return ``"date"`` for a ``date`` type and ``"key"`` for any other type.
    _version_mismatches: List the ways a table does not fit a ``Versioned`` writer's day columns and version key.

"""

import dataclasses
import math
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from types import NoneType, UnionType
from typing import Annotated, Any, Final, Literal, TypeAliasType, TypeVar, Union, get_args, get_origin
from uuid import UUID

import polars as pl
from annotated_types import Ge, Gt, Le, Lt
from pydantic import AwareDatetime, BaseModel
from pydantic.fields import FieldInfo

from pipeline.db.catalog import TableInfo

#: A ``schema.table`` name: two lower-case identifiers of ASCII letters, digits and ``_``, not starting with a digit,
#: joined by one dot. Used with ``fullmatch``. Each identifier is 1 to 63 characters, the most Postgres keeps
#: (NAMEDATALEN - 1), so a name is never truncated by Postgres.
TABLE_PATTERN: Final = re.compile(r"[a-z_][a-z0-9_]{0,62}\.[a-z_][a-z0-9_]{0,62}")
#: A bare column name: one lower-case identifier of ASCII letters, digits and ``_``, not starting with a digit, 1 to
#: 63 characters, the most Postgres keeps (NAMEDATALEN - 1). Used with ``fullmatch``.
COLUMN_PATTERN: Final = re.compile(r"[a-z_][a-z0-9_]{0,62}")

#: The Postgres type name of a ``timestamptz`` column, as ``format_type`` prints it.
_TIMESTAMPTZ: Final = "timestamp with time zone"
#: The Postgres types a ``Versioned`` writer's day columns may have: ``integer`` or ``bigint`` (``YYYYMMDD`` date keys)
#: or ``date``. ``_check_writer`` holds ``opens_from`` to them; ``_version_mismatches`` holds ``valid_from`` and
#: ``valid_to`` to them. ``smallint`` is left out: it cannot hold a ``YYYYMMDD`` key.
_VERSION_TYPES: Final = ("integer", "bigint", "date")
#: The frame type and Postgres type of each scalar annotation a ``To`` field may have; ``Decimal`` is mapped apart,
#: in ``_column_type``. A plain ``datetime`` is not mapped, only ``AwareDatetime``; ``UUID`` is a string in the frame.
_SCALARS: Final[dict[object, tuple[pl.DataType, str]]] = {
    str: (pl.String(), "text"),
    int: (pl.Int64(), "bigint"),
    AwareDatetime: (pl.Datetime("us", "UTC"), _TIMESTAMPTZ),  # "us": microseconds, the precision timestamptz stores
    date: (pl.Date(), "date"),
    bool: (pl.Boolean(), "boolean"),
    UUID: (pl.String(), "uuid"),
}
#: The largest ``max_digits`` a loaded ``Decimal`` may have: the largest precision ``pl.Decimal`` holds.
_MAX_DIGITS: Final = 38
#: The frame type and Postgres type of an ``int`` field with a ``Width`` mark, by its bits; without one, ``bigint``.
#: Its keys are also the widths ``Width.__post_init__`` accepts.
_INT_WIDTHS: Final[dict[int, tuple[pl.DataType, str]]] = {32: (pl.Int32(), "integer"), 16: (pl.Int16(), "smallint")}
#: The Postgres types an ``Update`` with ``missing="zero"`` can set to zero: ``bigint``, ``integer``, ``smallint`` (the
#: types ``_column_type`` gives an ``int``) or ``numeric(p,s)`` (a ``Decimal``). Used with ``fullmatch``.
_NUMERIC: Final = re.compile(r"bigint|integer|smallint|numeric\(\d+,\d+\)")


@dataclass(frozen=True, slots=True)
class To:
    """Field mark that loads a single value into the named column.

    ``__post_init__`` is not defined, so ``column`` is not validated here; ``_column`` checks it against
    ``COLUMN_PATTERN`` when the model's target is built.

    Attributes:
        column: The target column name.

    """

    column: str


@dataclass(frozen=True, slots=True)
class Width:
    """Field mark that loads an ``int`` field into a narrower integer column than the default ``bigint``.

    Used through ``pipeline.checks.record``'s ``Int32`` and ``Int16``, which carry it with the matching bounds so an
    out-of-range value fails Validate instead of the load. ``__post_init__`` refuses any width but 32 and 16 (the keys
    of ``_INT_WIDTHS``). The mark takes effect only on a ``To`` field, where ``_column_type`` refuses it unless it is
    the field's only ``Width``, the field is an ``int``, and the field's bounds lie within the width's range (so a
    bare ``Width``, or one nested over an ``Int32``, cannot let a value through Validate that the column refuses). On
    any other field it is ignored.

    Attributes:
        bits: 32 for an ``integer`` column, 16 for a ``smallint`` column.

    """

    bits: Literal[16, 32]

    def __post_init__(self) -> None:
        """Refuse a width other than 32 or 16.

        Preconditions:
            None.

        Returns:
            None.

        Raises:
            ValueError: ``bits`` is not 32 or 16.

        """
        if self.bits not in _INT_WIDTHS:
            raise ValueError(f"Width needs 32 or 16 bits, got {self.bits!r}")


@dataclass(frozen=True, slots=True)
class Rows:
    """Field mark that loads a child part, or a ``list`` of them, as rows of the child's own table.

    Only a field of the class that declares the table (or of a container record, ``build_container``) may carry it,
    never a field of a nested part without a table. The field is either ``list[SomePart]`` (one row per item; neither
    the field nor an item may be None) or a single ``SomePart`` (one-to-one: one row, or none when the field is
    ``SomePart | None`` and holds None). ``SomePart`` must declare a table with no source column and an ``Upsert``,
    ``Versioned`` or ``ReplaceChildren`` writer; an ``Upsert`` or ``Versioned`` item of a list needs a key of its own.
    ``_columns`` and ``_child`` check this. A child may carry ``Rows`` fields of its own, so children nest;
    ``build_target`` gives each child the columns it inherits from its parent (see ``Target``).

    """

    ...


@dataclass(frozen=True, slots=True)
class Upsert:
    """Writer that inserts rows or updates them on a conflict over ``key``.

    ``__post_init__`` is not defined, so ``key`` is not validated here; ``_check_writer`` checks it when the target is
    built. As a child's writer, ``key`` names the child's own columns, and the parent's columns it inherits are plain
    columns outside it unless ``key`` names them too: a child keyed by its parent's id and its own
    (``("request_id", "line_item_id")``, with ``request_id`` inherited) names both, and its table's unique key is that
    pair. An inherited column in ``key`` is checked when the parent adopts the child (``_adopt``), as the child's own
    class cannot know what it will inherit. A one-to-one child may leave ``key`` empty, and is then keyed by its
    inherited columns (``row_key``). An ``Upsert`` child never deletes: a row its parent no longer lists stays as it is.

    Attributes:
        key: Target column names: distinct, not nullable, and including the source column when the table has one;
            non-empty except for a one-to-one child's (``_child`` refuses an empty key for a list's items, and
            ``pipeline.checks.record`` refuses one for a ``Record``).

    """

    key: tuple[str, ...]  # target columns, the source column included when the table has one


@dataclass(frozen=True, slots=True)
class ReplaceChildren:
    """Writer for a child table whose rows are replaced under each parent's key, matched on ``key``.

    The child target inherits its parent's row key columns (a top-level parent's whole writer key, the source column
    included); ``key`` names the child's own columns that, after the inherited ones, identify one row (``row_key``).
    ``__post_init__`` is not defined, so ``key`` is not validated here; ``_check_writer`` checks it when the target is
    built. A target with this writer has no source column (``build_target`` refuses one) and is reached only through
    its parent's ``children``; its own children inherit its row key.

    Attributes:
        key: The child's own column names: non-empty, distinct, not nullable, none of them inherited.

    """

    key: tuple[str, ...]  # the child's own columns; the row key is the inherited columns, then these


@dataclass(frozen=True, slots=True)
class ReplacePartition:
    """Writer that replaces the rows whose ``column`` lies in the task's time frame.

    Only the task's source's rows are replaced when the table has a source column; otherwise every row in the frame.
    It has no key, so a target with this writer cannot have ``Rows`` children.

    ``__post_init__`` is not defined, so ``column`` is not validated here; ``_check_writer`` checks it is a target
    column of type ``timestamptz`` or ``date`` that does not allow None when the target is built.

    Attributes:
        column: The target column whose value places a row in the frame.

    """

    column: str  # a timestamptz or date column; rows with it in the task's frame are replaced


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Writer keyed by ``key`` that sets ``removed_at`` itself.

    ``__post_init__`` is not defined, so nothing is validated here; ``_check_writer`` checks ``key`` and that no field
    loads into ``removed_at`` when the target is built, and ``_child`` refuses ``prune=False`` on a ``Versioned`` child.

    A top-level target is pruned only when ``prune`` is true: a load that holds no record then marks the source's rows
    missing from its frame removed. With ``prune=False`` it is never pruned, so a target fed by an update window (only
    the rows changed in the window are fetched) keeps the rows the window leaves out, and an empty frame is not
    refused, since it removes nothing. A pruned target's empty frame is refused (``NothingLoaded``), since it would
    remove every row of the source, unless ``allow_empty`` is true: the connector then leaves out what does not exist
    (TikTok omits goods with no stock), so an empty frame is a valid day on which nothing exists, and the prune runs
    and removes every row of the source. ``_check_writer`` refuses ``allow_empty`` without ``prune``, which would have
    no effect, and ``_child`` refuses it on a ``Versioned`` child, whose empty frame is always valid.

    Attributes:
        key: Target column names: non-empty, distinct, not nullable, and including the source column when the table has
            one.
        prune: Keyword-only. Whether a top-level target's rows missing from its frame are pruned; only ``True``, the
            default, for a child, which is pruned under its staged parents.
        allow_empty: Keyword-only. Whether a top-level target's empty frame is valid and prunes every row of the
            source; ``False`` by default, and always for a child. Needs ``prune``.

    """

    key: tuple[str, ...]  # target columns, the source column included when the table has one
    prune: bool = dataclasses.field(default=True, kw_only=True)  # False: a top-level target is never pruned
    allow_empty: bool = dataclasses.field(default=False, kw_only=True)  # True: an empty frame prunes every row


@dataclass(frozen=True, slots=True)
class Update:
    """Writer that updates existing rows matched on ``key``; ``missing`` says what happens to rows not returned.

    ``__post_init__`` is not defined, so nothing is validated here; ``_check_writer`` checks, when the target is built,
    ``key``, that some column lies outside it, and for ``missing="zero"`` that every column outside it is numeric.

    Attributes:
        key: Target column names: non-empty, distinct, not nullable, and including the source column when the table has
            one.
        missing: ``"keep"`` leaves the source's rows the connector did not return (every table row when the table has
            no source column); ``"zero"`` sets their columns outside ``key`` to zero.

    """

    key: tuple[str, ...]  # target columns, the source column included when the table has one
    missing: Literal["keep", "zero"] = "keep"  # what happens to the source's rows the connector did not return


@dataclass(frozen=True, slots=True)
class Versioned(Snapshot):
    """A ``Snapshot`` that keeps history: each id has versions, and a change closes the open one and opens another.

    The writer owns ``valid_from`` and ``valid_to``: each version's first and last day (inclusive; ``valid_to`` is
    None while the version is open). On the task's observation day D (``load``'s ``zone`` date of the task's
    ``claimed_at``), for each staged id: a new id opens a version from its ``opens_from`` value (or D when
    ``opens_from`` is None or the value is null); an id whose open version's other columns equal the staged row is
    left alone; a changed id has its open version closed on D - 1 and a new one opened from D, unless the open version
    already started on D, which is overwritten in place; an id whose versions are all closed opens a new one from D;
    an id with a version that starts after D, or was closed on D or later, was written by a later day's task, so the
    observation is stale: left alone and counted. With pruning, an open version of the source whose id is not staged
    is closed on D - 1, or deleted when it started on D (the day's last observation wins). Nothing else is deleted.

    As a ``Snapshot`` subclass it shares ``Snapshot``'s rules wherever they test for ``Snapshot``: ``key`` is checked
    as a writer key (distinct, not nullable, the source column included), and for a top-level target ``prune``
    decides whether missing ids are closed at all, no missing id is closed while records are held, and an empty frame
    is refused when pruning unless ``allow_empty`` is set, when it closes every open version of the source.
    ``_check_writer`` and ``mismatches`` give it checks of its own in place of ``Snapshot``'s ``removed_at`` ones.

    As a child's writer, the id is the child's row key (``row_key``): its own ``key``, or, for a one-to-one child that
    leaves ``key`` empty, the columns it inherits from its parent, which are otherwise plain columns. A child is always
    pruned, but only under its parents staged in the same load: an open version whose inherited columns match a staged
    parent and whose id is not staged is closed on D - 1, or deleted when it started on D, even while records are held
    (a held record's parent is not staged). An empty child frame is valid. Closing a parent closes none of its
    children. A child keeps ``prune`` and ``allow_empty`` at their defaults (``_child`` refuses either changed), as
    its scoped prune is what keeps it equal to its parent's list.

    Attributes:
        key: As for ``Snapshot``: the columns that identify an id, the source column included when the table has one;
            for a child, its own columns, empty only for a one-to-one child.
        prune: Keyword-only, inherited from ``Snapshot``: whether a top-level target's missing ids are closed;
            ``True`` by default, and always for a child.
        allow_empty: Keyword-only, inherited from ``Snapshot``: whether a top-level target's empty frame is valid and
            closes every open version of the source; ``False`` by default, and always for a child.
        valid_from: The table column holding a version's first day; ``integer`` or ``bigint`` (``YYYYMMDD`` date keys)
            or ``date``, not nullable, set by the writer only.
        valid_to: The table column holding a version's last day, of ``valid_from``'s kind (a date key of either
            width, or a date), nullable (None while the version is open), set by the writer only.
        opens_from: A target column of an ``int`` field without a 16-bit ``Width``, or of a ``date`` field, of
            ``valid_from``'s kind, dating a new id's first version; or None to open it on D. For a child, one of its
            own columns, never an inherited one (``_check_writer`` sees only the child's own columns).

    """

    valid_from: str  # the version's first day; set by the writer
    valid_to: str  # the version's last day, inclusive; NULL while open; set by the writer
    opens_from: str | None = None  # a target column dating a new id's first version, or None for the observation day


#: The writer of a target: how its frame is written to its table. ``Versioned`` is a ``Snapshot`` subclass, so a
#: ``match`` or ``isinstance`` test that must tell them apart tests ``Versioned`` first.
type Writer = Upsert | ReplaceChildren | ReplacePartition | Snapshot | Versioned | Update


@dataclass(frozen=True, slots=True)
class Column:
    """One target column: its name, frame and Postgres types, nullability and path on the record.

    ``__post_init__`` is not defined, so nothing is validated here; ``build_target`` makes every instance.

    Attributes:
        name: The target column name, matching ``COLUMN_PATTERN``.
        polars: The column's type in the frame.
        postgres: The column's type exactly as ``format_type`` prints it.
        nullable: True when the field allows None, or lies inside a part that does.
        path: Field names from the record to the value, nested parts included; ``()`` for the source column. In a child
            target, the child's own columns' paths are on the child part, and an inherited column is the ``Column`` of
            the target that first defined it, so its path is on that target's model: the parent, or, for a column
            inherited through several levels (a ``ReplaceChildren`` child passes on what it inherited), an ancestor
            further up.

    """

    name: str  # target column
    polars: pl.DataType  # frame type
    postgres: str  # exactly as Postgres format_type() prints it
    nullable: bool
    path: tuple[
        str, ...
    ]  # where the value is on the record: field names, nested parts included; () for the source column


@dataclass(frozen=True, slots=True)
class Target:
    """A table, its writer, source column, columns and child targets; or a container's key columns and children.

    ``__post_init__`` is not defined, so nothing is validated here; every instance comes from ``build_target`` or
    ``build_container`` (or ``_adopt``, which both call), which check the names, the writer against the columns, and
    that column names are distinct.

    A container (``build_container``) is a top-level target with no table and no writer: ``table`` and ``writer`` are
    both None, and for no other target is either None. Its columns are its source column, when it has one, and its
    ``record_key`` fields' columns: its key, which Load stages for its children but writes nowhere.

    A child target (one in some target's ``children``) has a non-empty ``inherited``; a top-level target has none. What
    a child inherits, under the parent's own column names: from a top-level parent with a table, its writer key,
    without the source column for an ``Upsert`` or ``Versioned`` child, with it for a ``ReplaceChildren`` child; from
    a container, every one of its columns (the source column included), whatever the child's writer; from a child
    parent, that parent's row key (``row_key``), which never holds a source column.

    Attributes:
        table: The table as ``schema.table``, matching ``TABLE_PATTERN``; None for a container.
        writer: How the frame is written to ``table``; None for a container.
        source_column: The column the task's source id is written to, or None; always None for a child.
        columns: Every column the frame has, in frame order: the source column first, then inherited, then own.
        inherited: The column names copied from the parent row (see above); ``()`` for a top-level target.
        children: ``(Rows field name, child target)`` pairs in field order; children may have children of their own.

    """

    table: str | None  # "schema.table"; None for a container
    writer: Writer | None  # None for a container
    source_column: str | None
    columns: tuple[
        Column, ...
    ]  # every column the frame has, in frame order: source column first, then inherited, then own
    inherited: tuple[str, ...]  # columns copied from the parent row; () for a top-level target, never () for a child
    children: tuple[tuple[str, Target], ...]  # (the Rows field name, the child's target), in field order


def target_of(model: type[BaseModel]) -> Target:
    """Return a model's top-level target.

    Only a ``Record`` loads at the top level, as a target with a table or as a container (``build_container``).
    ``pipeline.checks.record`` imports this module, so a ``Record`` is told by its ``record_key``, which every
    ``Record`` subclass must set (and which a ``Part`` does not have), rather than by ``issubclass``. A ``Part`` with a
    table loads only as a child, whatever its writer, so its target here is not the full one: that is in its parent's
    ``children``, with its inherited columns.

    Preconditions:
        None.

    Args:
        model: The record model; its ``__target__`` and ``record_key`` attributes are read, and a missing attribute
            counts as None.

    Returns:
        ``model.__target__``.

    Raises:
        TypeError: ``model`` has no target (it declares no table and is not a container), has no ``record_key`` (a
            ``Part`` rather than a ``Record``), or its writer is ``ReplaceChildren``; in the last two cases it loads as
            child rows.

    """
    target: Target | None = getattr(model, "__target__", None)
    if target is None:
        raise TypeError(f"{model.__name__} declares no table")
    if getattr(model, "record_key", None) is None or isinstance(target.writer, ReplaceChildren):
        raise TypeError(f"{model.__name__} loads as child rows; its target is in its parent's children")
    return target


def walk(target: Target) -> Iterator[Target]:
    """Yield a target and every target under it, parents first.

    The order is pre-order: ``target``, then for each child in field order the child and everything under it, so a
    target always comes before its children and a parent's whole subtree before its next sibling.

    Preconditions:
        None.

    Args:
        target: The target to start from.

    Yields:
        ``target``, then each target under it, in the order above.

    Raises:
        Nothing.

    """
    yield target
    for _, child in target.children:
        yield from walk(child)


def build_target(model: type[BaseModel], table: str, writer: Writer, source_column: str | None) -> Target:
    """Build and check a model's target from its table, writer, source column and field marks.

    The columns are the source column (``uuid``, not nullable, path ``()``) when given, then the columns ``_columns``
    collects. Each ``Rows`` field gives a child, whose target (as its own class built it, with its own children) is
    adopted by ``_adopt``: rebuilt with the columns it inherits before its own, and so on down its children. What a
    child inherits is this target's row key (``row_key``: the writer key for a top-level target), less the source
    column for an ``Upsert`` or ``Versioned`` child; a ``ReplaceChildren`` child inherits it whole. So a target needs
    a non-empty row key to have children: a ``ReplacePartition`` target cannot have any, and nor can a one-to-one
    child that leaves its key empty (its row key is known only once its parent adopts it). An ``Update`` parent
    cannot have children either: the row it updates may not exist, so its child rows would have no parent. Every
    table in the finished tree (this target's and every child's, at any depth) must differ.

    What each child is passed is also checked to be non-empty: with a writer key of the source column alone, an
    ``Upsert`` or ``Versioned`` child would be passed nothing once the source column is left out, and so could not
    tell which parent row it belongs to, so ``Rows`` is refused there too. A ``Record`` cannot reach this
    (``pipeline.checks.record`` refuses a writer key that lacks its ``record_key`` columns), but a ``Part`` declaring
    a table with a source column, or a direct call, can.

    Preconditions:
        None.

    Args:
        model: The model declaring the table.
        table: The table as ``schema.table`` in lower case.
        writer: How rows are written to ``table``.
        source_column: The column the task's source id is written to, or None.

    Returns:
        The target, with ``inherited=()`` and one child per ``Rows`` field in field order, each adopted with its
        inherited columns.

    Raises:
        ValueError: ``table`` or ``source_column`` is not a lower-case name of the right form (an identifier over 63
            characters included); from ``_columns``, for a malformed ``To`` column or a ``Decimal`` precision or scale
            out of range; from ``_check_writer``, for a writer that does not fit the columns.
        TypeError: ``source_column`` is given with ``ReplaceChildren``; ``model`` is not complete; ``Rows`` is used with
            a target whose row key is empty or None, or that passes a child no column (a writer key of the source
            column alone, for an ``Upsert`` or ``Versioned`` child), or with ``Update``; two targets in the tree load
            into one table;
            from ``_columns``, for a misused mark (``Rows`` on a ``list`` without exactly one argument, or on a part
            whose table does not fit a child, included; a ``Width`` on a ``To`` field that is not an ``int``, is one
            of several or is unbounded for its width), a type with no column type, a part that contains itself, an
            unmarked union with a part among its members or a generic type alias whose arguments cannot be
            substituted; from ``_check_names`` (directly or through ``_adopt``), for a repeated column name in the
            model or in a child at any depth; from ``_check_writer``, for a ``ReplaceChildren`` model with no column of
            its own.

    """
    name = model.__name__
    if not TABLE_PATTERN.fullmatch(table):
        raise ValueError(f"{name}: table must be schema.table in lower case, got {table!r}")
    if source_column is not None and not COLUMN_PATTERN.fullmatch(source_column):
        raise ValueError(f"{name}: source_column must be a lower-case name, got {source_column!r}")
    if source_column is not None and isinstance(writer, ReplaceChildren):
        raise TypeError(f"{name}: a ReplaceChildren table has no source_column; the source comes with the parent's key")
    if not model.__pydantic_complete__:
        raise TypeError(f"{name} declares a table, so its field types must be defined first")
    source = (Column(source_column, pl.String(), "uuid", False, ()),) if source_column is not None else ()
    rows: list[tuple[str, Target]] = []
    columns = (*source, *_columns(model, (), False, rows, (model,)))
    _check_names(name, columns)
    part = getattr(model, "record_key", None) is None  # a Part loads only as a child, so it may inherit key columns
    _check_writer(name, writer, columns, source_column, part)
    key = row_key(Target(table, writer, source_column, columns, (), ()))  # the row key, as a child would see it
    if rows and not key:
        raise TypeError(f"{name}: Rows needs a writer with a key of its own, such as Upsert or ReplaceChildren")
    if rows and isinstance(writer, Update):
        raise TypeError(f"{name}: Rows is not for an Update table; its rows may not exist to own the children")
    by_name = {column.name: column for column in columns}
    children: list[tuple[str, Target]] = []
    for field, child in rows:  # a keyed child is per parent id, not per source: it leaves the source column out
        whole = isinstance(child.writer, ReplaceChildren)
        passed = tuple(  # a key column this Part inherits is not known yet: _adopt passes it on once it is
            by_name[column] for column in key or () if (whole or column != source_column) and column in by_name
        )
        if not passed:
            raise TypeError(
                f"{name}.{field}: Rows needs its parent's key to hold a column besides the source column "
                f"{source_column!r}, for the child rows to name their parent"
            )
        children.append((field, _adopt(f"{name}.{field}", child, passed)))
    target = Target(table, writer, source_column, columns, (), tuple(children))
    _check_tables(name, target)
    return target


def build_container(model: type[BaseModel], key: Sequence[str], source_column: str | None) -> Target:
    """Build and check a container record's target: no table, only its key columns and ``Rows`` children.

    A container is a record that fills only its children's tables. Its columns are the source column (``uuid``, not
    nullable, path ``()``) when given, then the columns ``_columns`` collects, which must be exactly one ``To``
    column for each ``key`` field: those are what its children inherit, so any other ``To`` (a computed field's
    included) would load nowhere and is refused. It needs at least one ``Rows`` field. Each child is adopted by
    ``_adopt`` with every container column, the source column included, whatever its writer, and is then keyed by the
    usual child rules (``row_key``). Every table in the tree must differ.

    Preconditions:
        ``key`` is the model's checked ``record_key`` (``pipeline.checks.record`` calls this after checking it).

    Args:
        model: The container record model.
        key: Its ``record_key``: the fields whose columns its children inherit.
        source_column: The column the task's source id is written to, or None.

    Returns:
        The container's target: ``table`` and ``writer`` None, ``inherited=()``, and one child per ``Rows`` field in
        field order, each adopted with the container's columns.

    Raises:
        ValueError: ``source_column`` is not a lower-case name of the right form; from ``_columns``, for a malformed
            ``To`` column or a ``Decimal`` precision or scale out of range.
        TypeError: ``model`` is not complete; a ``key`` field has no ``To``; a field other than a ``key`` field has a
            ``To``; there is no ``Rows`` field; two targets in the tree load into one table; from ``_columns``, for a
            misused mark or field type as for ``build_target``; from ``_check_names`` (directly or through
            ``_adopt``), for a repeated column name in the model or in a child at any depth.

    """
    name = model.__name__
    if source_column is not None and not COLUMN_PATTERN.fullmatch(source_column):
        raise ValueError(f"{name}: source_column must be a lower-case name, got {source_column!r}")
    if not model.__pydantic_complete__:
        raise TypeError(f"{name} is a container, so its field types must be defined first")
    source = (Column(source_column, pl.String(), "uuid", False, ()),) if source_column is not None else ()
    rows: list[tuple[str, Target]] = []
    columns = (*source, *_columns(model, (), False, rows, (model,)))
    _check_names(name, columns)
    loaded = {column.path for column in columns if column.path}  # every column but the source column
    for field in key:
        if (field,) not in loaded:
            raise TypeError(f"{name}.{field} is in record_key, so it needs a To")
    loose = sorted(column.name for column in columns if column.path and column.path not in {(f,) for f in key})
    if loose:
        raise TypeError(f"{name}: a container loads no table, so only record_key fields take a To: {', '.join(loose)}")
    if not rows:
        raise TypeError(f"{name}: a container loads only its Rows children, so it needs one")
    children = tuple((field, _adopt(f"{name}.{field}", child, columns)) for field, child in rows)
    target = Target(None, None, source_column, columns, (), children)
    _check_tables(name, target)
    return target


def _check_tables(name: str, target: Target) -> None:
    """Check that no two targets in a tree load into one table.

    A container has no table, so it is left out of the comparison.

    Preconditions:
        None.

    Args:
        name: The model name, used in the error message.
        target: The top of the tree; every target ``walk`` yields from it is compared.

    Returns:
        None.

    Raises:
        TypeError: Two targets load into one table; the message lists each such table once, sorted.

    """
    tables = [found.table for found in walk(target) if found.table is not None]
    shared = sorted({t for t in tables if tables.count(t) > 1})
    if shared:
        raise TypeError(f"{name}: more than one target loads into {', '.join(shared)}")


def _adopt(where: str, child: Target, inherited: tuple[Column, ...]) -> Target:
    """Rebuild a child target, and every target under it, with the columns it inherits from its parent.

    The child's own columns are its columns less those it inherited before (``child.inherited``: ``()`` as its own
    class built it, or the columns of an earlier parent's adoption); the new columns are ``inherited`` followed by
    them. Each of the child's own children is then adopted in turn, inheriting the rebuilt child's row key
    (``row_key``), whose columns keep their names. So a whole subtree takes its inherited columns from the top
    parent that adopts it last.

    Preconditions:
        ``child`` came from ``build_target`` (directly or through an earlier ``_adopt``) and has no source column
        (``_child`` refuses one); ``inherited`` is non-empty (``build_target`` refuses ``Rows`` otherwise, a container
        passes its key columns, and a child parent's row key is never empty), and its columns are what the parent
        passes: a top-level parent's row key, less the source column for an ``Upsert`` or ``Versioned`` child
        (``build_target``); every container column (``build_container``); or a child parent's row key (this
        function, one level up).

    Args:
        where: ``Model.field`` (``Model.field.field`` and so on below), used in error messages.
        child: The child target to rebuild.
        inherited: The parent's columns the child inherits, in the parent's column or row key order.

    Returns:
        The rebuilt target: ``columns`` the inherited columns then the child's own, ``inherited`` their names, and
        every child adopted the same way.

    Raises:
        TypeError: From ``_check_names``, when an inherited column has the name of one of the child's own columns, at
            this depth or below.
        ValueError: An ``Upsert`` child's key names a column that is neither its own nor inherited, at this depth or
            below.

    """
    names = tuple(column.name for column in inherited)
    own = tuple(column for column in child.columns if column.name not in child.inherited)
    columns = (*inherited, *own)
    _check_names(where, columns)
    adopted = Target(child.table, child.writer, None, columns, names, ())
    by_name = {column.name: column for column in columns}
    own_key = _key(child.writer) if child.writer is not None else None  # a child always has a writer
    for column in own_key or ():  # an Upsert child's key may name inherited columns: now they are known
        if column not in by_name:
            raise ValueError(f"{where}: the writer's key names {column!r}, which the child neither has nor inherits")
    passed = tuple(by_name[column] for column in row_key(adopted) or ())  # never empty: own key or the inherited
    grandchildren = tuple((field, _adopt(f"{where}.{field}", grand, passed)) for field, grand in child.children)
    return Target(child.table, child.writer, None, columns, names, grandchildren)


def _columns(
    model: type[BaseModel],
    path: tuple[str, ...],
    optional: bool,
    rows: list[tuple[str, Target]] | None,
    inside: tuple[type[BaseModel], ...],  # the models this walk is in, so a part that contains itself is refused
) -> list[Column]:
    """Collect the columns of a model's marked fields and computed fields, nested parts included.

    Fields come first, in field order, then computed fields. A ``To`` field gives one column. An unmarked field whose
    type is a part gives the part's columns in its place; a computed field that returns a part does not. An unmarked
    field whose type is a union with a part among its members (``A | B``, ``A | B | None``, ``A | int``) is refused,
    since no one part gives its columns; ``A | None`` is a single optional part. A ``Rows`` field, on a list of parts
    or on a single part (``A`` or ``A | None``, one-to-one), appends ``(field name, child target)`` to ``rows`` and
    gives no column. Other unmarked fields, unmarked lists included, are skipped.

    Preconditions:
        None.

    Args:
        model: The model or nested part whose fields are read.
        path: Field names from the record to ``model``; ``()`` for the record itself.
        optional: True when ``model`` lies inside a part that allows None, so every column it gives is nullable.
        rows: The list ``Rows`` children are appended to, or None for a nested part, where ``Rows`` is refused.
        inside: The models this walk is in, ``model`` included, outermost first.

    Returns:
        The columns, in the order above.

    Raises:
        TypeError: ``To`` on a list; ``Rows`` on a nested part's field, on a field that is neither a list nor a part,
            or on a computed field; a ``To`` on a part field; an unmarked part field whose part has its own table; an
            unmarked part field whose part contains itself (its part is in ``inside``); an unmarked field whose type
            is a union with a part among its members; from ``_unwrap``, ``_mark``, ``_child`` and ``_column``.
        ValueError: From ``_column``, for a malformed ``To`` column or a ``Decimal`` precision or scale out of range.

    """
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
            if isinstance(mark, Rows):  # a one-to-one child
                if rows is None:
                    raise TypeError(f"{where}: Rows is only for a field of the class that declares the table")
                rows.append((field_name, _child(where, annotation, nullable)))
                continue
            if mark is not None:
                raise TypeError(f"{where}: {type(mark).__name__} is not for a part; mark the part's own fields")
            if getattr(annotation, "__target__", None) is not None:
                raise TypeError(f"{where}: {annotation.__name__} has its own table, so it loads only with Rows")
            if annotation in inside:
                raise TypeError(
                    f"{where}: {annotation.__name__} contains itself, so it cannot be flattened into columns"
                )
            columns.extend(_columns(annotation, here, optional or nullable, None, (*inside, annotation)))
        elif (
            mark is None
            and get_origin(annotation) in (Union, UnionType)
            and any(
                isinstance(member, type) and issubclass(member, BaseModel)
                for member in (_unwrap(arg, ())[0] for arg in get_args(annotation))
            )
        ):
            raise TypeError(f"{where}: a union of parts cannot be flattened into columns")
        elif isinstance(mark, Rows):
            raise TypeError(f"{where}: Rows is only for a part or a list of parts")
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
    """Strip type aliases, ``Annotated`` and ``Optional`` from an annotation.

    Layers are stripped from the outside in until none is left. A ``type`` alias, bare or subscripted
    (``Maybe[Address]`` for ``type Maybe[T] = T | None``), is replaced by ``alias_value``. Each ``Annotated`` adds its
    extras to the metadata; a ``FieldInfo`` extra adds its ``metadata`` items rather than itself. ``X | None`` becomes
    ``X`` and sets nullable. A union of two or more types besides None is returned whole, still holding None, with
    nullable set; a union without None (``int | str``) is returned whole with nullable as found so far.

    The metadata comes back innermost layer first: each stripped layer's items, in their own order, are placed
    before the ones already found, so ``metadata`` comes last. Pydantic applies an outer layer's constraints after an
    inner one's, so the last ``max_digits``/``decimal_places`` is the one it enforces: for a field such as
    ``Annotated[Money | None, Field(max_digits=12, decimal_places=2), To("price")]`` with
    ``type Money = Annotated[Decimal, Field(max_digits=10, decimal_places=2)]`` both the validator and
    ``_column_type`` use 12 digits.

    Preconditions:
        None.

    Args:
        annotation: The annotation to strip.
        metadata: Metadata already found (the outermost layer's), placed last in the result.

    Returns:
        The stripped annotation, the metadata found followed by ``metadata``, and whether None is allowed.

    Raises:
        TypeError: From ``alias_value``, for a subscripted generic alias whose arguments cannot be substituted.

    """
    found = list(metadata)
    nullable = False
    while True:
        if isinstance(annotation, TypeAliasType) or isinstance(get_origin(annotation), TypeAliasType):
            annotation = alias_value(annotation)  # type X = ..., bare or subscripted
        elif get_origin(annotation) is Annotated:
            annotation, *extra = get_args(annotation)
            layer = [part for item in extra for part in (item.metadata if isinstance(item, FieldInfo) else [item])]
            found[:0] = layer  # inner layers first: pydantic applies the outer layer's constraints last
        elif get_origin(annotation) in (Union, UnionType) and NoneType in get_args(annotation):
            rest = [arg for arg in get_args(annotation) if arg is not NoneType]
            if len(rest) > 1:
                return annotation, tuple(found), True  # a union of several types; refused if it is loaded
            annotation, nullable = rest[0], True
        else:
            return annotation, tuple(found), nullable


def alias_value(form: Any) -> Any:
    """Return a ``type`` alias's value, with a subscripted generic alias's arguments substituted.

    A bare alias (``type Code = str``) gives its ``__value__``. A subscripted generic alias (``Pair[int]`` for
    ``type Pair[T] = tuple[T, T]``) gives its alias's value with each type parameter replaced by its argument:
    ``tuple[int, int]``. The arguments are substituted only when the value is one of the alias's type variables
    (``type Id[T] = T``) or when the value's own parameters are the alias's type parameters in the same order; any
    other value (e.g. ``type Swap[A, B] = tuple[B, A]``, whose value lists its parameters in another order) is
    refused, since subscripting it would bind the arguments to the wrong parameters. Aliases inside the value are not
    resolved.

    Preconditions:
        ``form`` is a ``TypeAliasType`` or a subscripted one (its ``get_origin`` is a ``TypeAliasType``).

    Args:
        form: The alias, bare or subscripted.

    Returns:
        The alias's value, its arguments substituted.

    Raises:
        TypeError: ``form`` is a subscripted alias whose value is neither one of its type variables nor generic in
            its type parameters in order, or whose arguments do not fit its parameters.

    """
    if isinstance(form, TypeAliasType):
        return form.__value__
    alias: TypeAliasType = get_origin(form)
    params, value, args = alias.__type_params__, alias.__value__, get_args(form)
    if isinstance(value, TypeVar) and value in params and len(args) == len(params):
        return args[params.index(value)]
    if params and getattr(value, "__parameters__", ()) == params:
        return value[args]  # raises TypeError when the arguments do not fit
    raise TypeError(f"{form}: the arguments of a generic type alias are substituted only in its parameters' order")


def _mark(where: str, metadata: Sequence[object]) -> To | Rows | None:
    """Return the one ``To`` or ``Rows`` in a field's metadata, or None.

    Preconditions:
        None.

    Args:
        where: ``Model.field``, used in the error message.
        metadata: The field's metadata, from ``_unwrap``.

    Returns:
        The only ``To`` or ``Rows`` item, or None when there is none.

    Raises:
        TypeError: ``metadata`` holds more than one ``To`` or ``Rows``.

    """
    marks = [item for item in metadata if isinstance(item, To | Rows)]
    if len(marks) > 1:
        raise TypeError(f"{where} has more than one To or Rows")
    return marks[0] if marks else None


def _child(where: str, annotation: Any, nullable: bool) -> Target:
    """Return the target of a ``Rows`` field's child part, or of its list's item part.

    For a list, its item type ``X`` is passed through ``_unwrap`` first, as every other annotation is, so an item
    given as a ``type`` alias (``list[ChildAlias]`` for ``type ChildAlias = Child``) or wrapped in ``Annotated`` is
    read as the part it names; an item that allows None (``list[Child | None]``), and a list field that allows None,
    are refused. A single part ``X`` (already stripped by ``_columns``) is one-to-one and may allow None. Either way
    ``X`` must declare a table with no source column and an ``Upsert``, ``Versioned`` or ``ReplaceChildren`` writer
    (``Versioned`` is told apart from the ``Snapshot`` it subclasses, which is refused, as are ``Update`` and
    ``ReplacePartition``). A list's ``Upsert`` or ``Versioned`` item needs a non-empty key of its own: with none,
    every item of a parent would have the same key. A ``Versioned`` child may not set ``prune=False`` or
    ``allow_empty=True``: a child is always pruned under its staged parents, and its empty frame is always valid.

    Preconditions:
        ``annotation`` is a stripped annotation whose ``get_origin`` is ``list`` or that is a ``BaseModel`` subclass
        (``_columns`` calls this only then); a list may have any number of arguments, since pydantic accepts
        ``list[X, Y]`` as a field type.

    Args:
        where: ``Model.field``, used in the error message.
        annotation: The field's stripped annotation: ``list[X]`` or ``X``.
        nullable: Whether the field allows None.

    Returns:
        ``X.__target__``, whose writer is ``Upsert``, ``Versioned`` or ``ReplaceChildren`` and whose source column is
        None.

    Raises:
        TypeError: A list field or its item allows None, a list does not have exactly one argument (``list[X, Y]``),
            ``X`` declares no table, ``X`` has a source column, ``X``'s writer is not one of the three above, a
            list's ``X`` has an ``Upsert`` or ``Versioned`` writer with an empty key, or ``X``'s ``Versioned`` writer
            sets ``prune=False`` or ``allow_empty=True``; also from ``_unwrap``, for an item that is a subscripted
            generic alias whose arguments cannot be substituted.

    """
    target: Target | None = None
    many = get_origin(annotation) is list
    if not many:  # a single part: one-to-one, None allowed
        target = getattr(annotation, "__target__", None)
    elif len(get_args(annotation)) == 1 and not nullable:  # list[X] only
        item, _, item_nullable = _unwrap(get_args(annotation)[0], ())  # an alias or Annotated around the part
        target = None if item_nullable else getattr(item, "__target__", None)
    shape = "list[SomePart]" if many else "SomePart or SomePart | None"
    if target is None:
        raise TypeError(f"{where}: Rows needs {shape}, where SomePart declares a table")
    writer = target.writer
    if not isinstance(writer, Versioned | Upsert | ReplaceChildren):
        raise TypeError(f"{where}: a Rows child's writer must be Upsert, Versioned or ReplaceChildren")
    if target.source_column is not None:
        raise TypeError(f"{where}: a Rows child has no source_column; it inherits its parent's key")
    if many and isinstance(writer, Versioned | Upsert) and not writer.key:
        raise TypeError(f"{where}: a list's items need a key of their own; only a one-to-one child may have none")
    if isinstance(writer, Versioned) and not writer.prune:
        raise TypeError(
            f"{where}: a Rows child is always pruned under its staged parents, so it cannot set prune=False"
        )
    if isinstance(writer, Versioned) and writer.allow_empty:
        raise TypeError(f"{where}: a Rows child's empty frame is always valid, so it does not set allow_empty")
    return target


def _column(
    where: str, mark: To, annotation: Any, metadata: Sequence[object], nullable: bool, path: tuple[str, ...]
) -> Column:
    """Build the ``Column`` for a ``To`` field.

    Preconditions:
        None.

    Args:
        where: ``Model.field``, used in error messages.
        mark: The field's ``To``.
        annotation: The field's stripped annotation.
        metadata: The field's metadata, read for ``Decimal`` constraints.
        nullable: Whether the column allows None.
        path: Field names from the record to the value.

    Returns:
        The column named ``mark.column``.

    Raises:
        ValueError: ``mark.column`` does not match ``COLUMN_PATTERN``; from ``_column_type``, for a ``Decimal``
            precision or scale out of range.
        TypeError: From ``_column_type``, for a type with no column type.

    """
    if not COLUMN_PATTERN.fullmatch(mark.column):
        raise ValueError(f"{where}: To column must be a lower-case name, got {mark.column!r}")
    polars, postgres = _column_type(where, annotation, metadata)
    return Column(mark.column, polars, postgres, nullable, path)


def _column_type(where: str, annotation: Any, metadata: Sequence[object]) -> tuple[pl.DataType, str]:
    """Map an annotation to its frame type and Postgres type.

    ``Decimal`` maps to ``pl.Decimal(p, s)`` and ``numeric(p,s)``, with ``p`` and ``s`` the last non-None
    ``max_digits`` and ``decimal_places`` found on the metadata items; ``p`` must lie from 1 to 38 (the precisions
    polars holds) and ``s`` from 0 to ``p``. An ``int`` with a ``Width`` mark maps to ``integer`` (32 bits) or
    ``smallint`` (16 bits) through ``_INT_WIDTHS``. The mark must be the field's only ``Width``, and the field's
    numeric ``Ge``/``Gt`` and ``Le``/``Lt`` bounds must put its values within the width's range, as ``Int32`` and
    ``Int16`` do, so every value Validate accepts fits the column. Any other annotation is looked up in ``_SCALARS``.

    Preconditions:
        None.

    Args:
        where: ``Model.field``, used in error messages.
        annotation: The stripped annotation.
        metadata: The field's metadata.

    Returns:
        The frame type and the Postgres type as ``format_type`` prints it.

    Raises:
        TypeError: A ``Decimal`` lacks ``max_digits`` or ``decimal_places``; a ``Width`` mark is on a field that is
            not an ``int``, is one of several, or lacks a lower or upper bound within its range; or the annotation is
            not in ``_SCALARS`` (an unhashable annotation included).
        ValueError: A ``Decimal``'s ``max_digits`` or ``decimal_places`` lies outside the ranges above.

    """
    widths = [item.bits for item in metadata if isinstance(item, Width)]
    if widths:
        if annotation is not int:
            raise TypeError(f"{where}: a Width mark is for an int field, not {annotation!r}")
        if len(widths) > 1:
            raise TypeError(f"{where}: a field takes one Width mark, got {len(widths)}")
        bits = widths[0]
        least, most = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        lows: list[int] = []
        highs: list[int] = []
        for item in metadata:  # only numeric bounds count; an int above Gt(n) is at least floor(n) + 1, and so on
            if isinstance(item, Ge) and isinstance(item.ge, int | float):
                lows.append(math.ceil(item.ge))
            elif isinstance(item, Gt) and isinstance(item.gt, int | float):
                lows.append(math.floor(item.gt) + 1)
            elif isinstance(item, Le) and isinstance(item.le, int | float):
                highs.append(math.floor(item.le))
            elif isinstance(item, Lt) and isinstance(item.lt, int | float):
                highs.append(math.ceil(item.lt) - 1)
        if not lows or not highs or max(lows) < least or min(highs) > most:
            raise TypeError(f"{where}: Width({bits}) needs bounds within {least} to {most}; use Int{bits}")
        return _INT_WIDTHS[bits]
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
        if not 1 <= digits <= _MAX_DIGITS or not 0 <= places <= digits:
            raise ValueError(
                f"{where}: a loaded Decimal needs max_digits from 1 to {_MAX_DIGITS} and decimal_places from 0 to "
                f"max_digits, got {digits} and {places}"
            )
        return pl.Decimal(digits, places), f"numeric({digits},{places})"
    try:
        return _SCALARS[annotation]
    except (KeyError, TypeError) as err:  # TypeError: an unhashable annotation
        raise TypeError(f"{where}: {annotation!r} has no column type; see the type mapping") from err


def _check_names(where: str, columns: Sequence[Column]) -> None:
    """Check that no column name appears more than once.

    Preconditions:
        None.

    Args:
        where: The model, or ``Model.field`` for a child, used in the error message.
        columns: The columns to check.

    Returns:
        None.

    Raises:
        TypeError: A name appears more than once; the message lists each repeated name once, sorted.

    """
    names = [column.name for column in columns]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise TypeError(f"{where}: columns named more than once: {', '.join(repeated)}")


def _check_writer(
    name: str, writer: Writer, columns: Sequence[Column], source_column: str | None, part: bool = False
) -> None:
    """Check that a writer fits the columns.

    A writer key must name distinct target columns that do not allow None, and include the source column when there is
    one. It must be non-empty, except an ``Upsert`` or ``Versioned`` key, which may be empty for a one-to-one child (so
    keyed by the columns it inherits): the model's own class cannot tell how a parent will use it, so ``_child``
    refuses an empty key for a list's items, ``pipeline.checks.record`` refuses one for a ``Record``, and the source
    column check refuses one for a target with a source column. ``ReplacePartition`` needs its column to be a target
    column of type ``timestamptz`` or ``date`` that does not allow None. ``Versioned`` needs distinct ``valid_from``
    and ``valid_to`` names that no field loads into, and an ``opens_from`` that is a target column whose type is in
    ``_VERSION_TYPES`` (``bigint``, ``integer`` or ``date``: an ``int`` field with no 16-bit ``Width``, or a ``date``
    field); it is matched before ``Snapshot``, whose check it does not take. ``Snapshot`` refuses a column named
    ``removed_at``. Either refuses ``allow_empty`` without ``prune``, since an empty frame that prunes nothing needs
    no allowing. ``Update`` needs a column outside its key, and with ``missing="zero"`` every such column must match
    ``_NUMERIC`` (``bigint``, ``integer``, ``smallint`` or ``numeric(p,s)``).
    ``ReplaceChildren`` needs at least one column: ``columns`` are the child's own, as ``build_target`` adds the
    inherited key columns only when it rebuilds the child target, and without one every list item would load as the
    same key-only row; its ``key`` is then checked as a writer key is, against those own columns (``_adopt`` adds the
    inherited columns later). An ``Upsert`` or
    ``Versioned`` child's key, and a ``Versioned`` child's ``opens_from``, are likewise checked against its own
    columns only, never the inherited ones, except that a ``Part``'s ``Upsert`` key may also name columns it does not
    have (``part``): they must be inherited, which ``_adopt`` checks, and inherited columns are never nullable (they
    are a row key's). ``Upsert`` has no check beyond the key's.

    Preconditions:
        Column names in ``columns`` are distinct (``build_target`` calls ``_check_names`` first).

    Args:
        name: The model name, used in error messages.
        writer: The writer to check.
        columns: The target's columns.
        source_column: The target's source column, or None.
        part: Whether the target is a ``Part``'s, which loads only as a child. Defaults to ``False``.

    Returns:
        None.

    Raises:
        TypeError: The writer is ``ReplaceChildren`` and ``columns`` is empty.
        ValueError: Any other check above fails.

    """
    by_name = {column.name: column for column in columns}
    if isinstance(writer, ReplaceChildren) and not columns:
        raise TypeError(f"{name}: a ReplaceChildren table needs a column of its own besides the parent's key")
    key = writer.key if isinstance(writer, ReplaceChildren) else _key(writer)  # a child's own key columns
    if key is not None:
        one_to_one = isinstance(writer, Upsert | Versioned)  # may be empty: a one-to-one child keyed by its parent's
        if (not key and not one_to_one) or len(set(key)) != len(key):
            raise ValueError(f"{name}: the writer's key must name distinct columns, got {key!r}")
        for column in key:
            if column not in by_name and part and isinstance(writer, Upsert):
                continue  # to be inherited: _adopt checks it
            if column not in by_name:
                raise ValueError(f"{name}: the writer's key names {column!r}, which is not a target column")
            if by_name[column].nullable:
                raise ValueError(f"{name}: the writer's key column {column!r} cannot allow None")
        if source_column is not None and source_column not in key:
            raise ValueError(f"{name}: the writer's key must include the source column {source_column!r}")
    if isinstance(writer, Snapshot) and writer.allow_empty and not writer.prune:
        raise ValueError(f"{name}: allow_empty lets an empty frame prune every row, so it needs prune")
    match writer:
        case ReplacePartition(column=column):
            if column not in by_name or by_name[column].postgres not in (_TIMESTAMPTZ, "date"):
                raise ValueError(f"{name}: ReplacePartition needs a target column of type {_TIMESTAMPTZ} or date")
            if by_name[column].nullable:
                raise ValueError(f"{name}: the ReplacePartition column {column!r} cannot allow None")
        case Versioned(valid_from=valid_from, valid_to=valid_to, opens_from=opens_from):
            if valid_from == valid_to or not all(COLUMN_PATTERN.fullmatch(c) for c in (valid_from, valid_to)):
                raise ValueError(f"{name}: valid_from and valid_to must be two distinct lower-case column names")
            loaded = sorted(c for c in (valid_from, valid_to) if c in by_name)
            if loaded:
                raise ValueError(
                    f"{name}: a Versioned writer sets {', '.join(loaded)} itself, so no field may load into it"
                )
            if opens_from is not None and (
                opens_from not in by_name or by_name[opens_from].postgres not in _VERSION_TYPES
            ):
                raise ValueError(
                    f"{name}: opens_from must be a target column of an int or date field, got {opens_from!r}"
                )
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
    """Return a writer's key, or None for a writer without one.

    Preconditions:
        None.

    Args:
        writer: The writer.

    Returns:
        ``writer.key`` for ``Upsert``, ``Snapshot`` and ``Update``; None for ``ReplaceChildren`` and
        ``ReplacePartition``.

    Raises:
        Nothing.

    """
    match writer:
        case Upsert(key=key) | Snapshot(key=key) | Update(key=key):
            return key
    return None


def row_key(target: Target) -> tuple[str, ...] | None:
    """Return the columns that identify one row of a target's table, or None for a writer without them.

    A top-level target's row key is its writer's key; a container's is all its columns (its source column, then its
    ``record_key`` fields' columns). A child's depends on its writer (see ``Target`` for what it inherits): a
    ``ReplaceChildren`` child's is its inherited columns followed by its own ``key``; an ``Upsert`` or ``Versioned``
    child's is its own ``key``, with its inherited columns outside it as plain columns, unless that ``key`` is empty
    (a one-to-one child), when its row key is its inherited columns.

    Preconditions:
        None.

    Args:
        target: The target, as ``build_target`` or ``build_container`` built it (a child as ``_adopt`` rebuilt it),
            or a provisional one those functions build to read a row key.

    Returns:
        The writer's ``key`` for ``Upsert``, ``Snapshot``, ``Versioned`` and ``Update``, except ``target.inherited``
        when that key is empty; for ``ReplaceChildren``, the inherited columns followed by the writer's ``key``; None
        for ``ReplacePartition``; every column name, in order, for a container. ``()`` only for an ``Upsert`` or
        ``Versioned`` part with an empty key read on its own, before a parent adopts it.

    Raises:
        Nothing.

    """
    if target.writer is None:  # a container: its columns are its key
        return tuple(column.name for column in target.columns)
    if isinstance(target.writer, ReplaceChildren):
        return (*target.inherited, *target.writer.key)
    key = _key(target.writer)
    return target.inherited if key == () else key  # a one-to-one child with no key of its own


def _day_kind(postgres: str) -> str:
    """Return ``"date"`` for a ``date`` type and ``"key"`` for any other type.

    Meant for the types in ``_VERSION_TYPES``, where any type but ``date`` is an ``integer`` or ``bigint`` ``YYYYMMDD``
    date key; any other type (``text``, say) also gives ``"key"``, and its callers report such a type on its own.

    Preconditions:
        None.

    Args:
        postgres: A Postgres type name.

    Returns:
        ``"date"`` or ``"key"``.

    Raises:
        Nothing.

    """
    return "date" if postgres == "date" else "key"


def _version_mismatches(target: Target, writer: Versioned, info: TableInfo) -> list[str]:
    """List the ways a table does not fit a ``Versioned`` writer's day columns and version key.

    Each day column gets at most one of: missing, not in ``_VERSION_TYPES``, not writable. When both exist, the two
    are compared by ``_day_kind`` (an ``integer`` and a ``bigint`` match), ``valid_from`` must be ``NOT NULL`` and
    ``valid_to`` nullable, and ``opens_from``, when set, must be of their kind. Last, a unique key must be on exactly
    the target's row key (``row_key``: ``writer.key``, or a one-to-one child's inherited columns when that is empty)
    and ``valid_from``, as ``mismatches`` requires for a row key: a narrower one would refuse versions the writer may
    write, a wider one would let two versions of a key share a day.

    Preconditions:
        ``info`` describes ``target.table``.

    Args:
        target: A target whose writer is ``writer``.
        writer: The target's ``Versioned`` writer.
        info: The table as the catalog describes it.

    Returns:
        One message per mismatch, in the order above; ``[]`` when it fits.

    Raises:
        Nothing.

    """
    table, found = target.table, []
    first, last = info.columns.get(writer.valid_from), info.columns.get(writer.valid_to)
    for column, actual in ((writer.valid_from, first), (writer.valid_to, last)):
        if actual is None:
            found.append(f"{table}.{column} does not exist")
        elif actual.type not in _VERSION_TYPES:
            found.append(f"{table}.{column} is {actual.type}, a Versioned writer needs integer, bigint or date")
        elif not actual.writable:
            found.append(f"{table}.{column} is generated, so the Versioned writer cannot set it")
    if first is not None and last is not None:
        if _day_kind(first.type) != _day_kind(last.type):
            found.append(f"{table}.{writer.valid_from} and {writer.valid_to} must both be date keys or both dates")
        if first.nullable:
            found.append(f"{table}.{writer.valid_from} must be NOT NULL")
        if not last.nullable:
            found.append(f"{table}.{writer.valid_to} must be nullable: an open version has none")
        opens = next((c for c in target.columns if c.name == writer.opens_from), None)
        if opens is not None and _day_kind(opens.postgres) != _day_kind(first.type):
            found.append(f"{table}.{writer.opens_from} must be a date key or a date as {writer.valid_from} is")
    versions = frozenset((*(row_key(target) or ()), writer.valid_from))  # never None for a Versioned target
    if versions not in info.unique_keys:
        found.append(f"{table} has no unique key on exactly {', '.join(sorted(versions))}")
    return found


def mismatches(target: Target, info: TableInfo | None) -> list[str]:
    """List the ways a table in the catalog does not fit a target.

    Each target column gets at most one message, the first that applies: it does not exist, its type differs from
    ``Column.postgres`` (compared as exact strings), it is ``NOT NULL`` while the model allows None, or it is
    not writable (a generated or ``GENERATED ALWAYS`` identity column), except an ``Update``'s key columns, which
    are only matched, never written. Then, unless the writer is ``Update`` (which never inserts), each table column
    that is ``NOT NULL`` with no default and not a target column gets a message, except ``removed_at`` for a
    ``Snapshot`` and ``valid_from`` for a ``Versioned``. A row key (``row_key``) needs a unique key on exactly its
    columns, and a ``Snapshot`` needs a nullable ``timestamptz`` ``removed_at`` that is writable, since the writer sets
    it. A ``Versioned`` table instead holds several versions per key, so ``_version_mismatches`` gives its checks:
    ``valid_from`` and ``valid_to`` of one kind (``integer`` or ``bigint`` date keys, mixed widths allowed, or both
    ``date``), writable, ``valid_from`` ``NOT NULL`` and ``valid_to`` nullable; ``opens_from``, when set, of that kind;
    and a unique key on exactly the row key and ``valid_from``. Only ``target.table`` is
    compared; ``target.children`` are not visited, so each child target is checked by its own call. A container has no
    table, so nothing is compared and ``info`` is not read.

    Preconditions:
        ``info`` is ``table_info``'s result for ``target.table`` (any value for a container).

    Args:
        target: The target to check.
        info: The table as the catalog describes it, or None when ``table_info`` found no table.

    Returns:
        One message per mismatch, in the order above; ``[]`` when the table fits, and always for a container. A
        single message when ``info`` is None.

    Raises:
        Nothing.

    """
    table = target.table
    if table is None or target.writer is None:  # a container: no table to compare
        return []
    if info is None:
        return [f"{table} does not exist or has 0 columns"]
    found: list[str] = []
    names = {column.name for column in target.columns}
    update = isinstance(target.writer, Update)
    matched = (_key(target.writer) or ()) if update else ()  # an Update never writes its key
    for column in target.columns:
        actual = info.columns.get(column.name)
        if actual is None:
            found.append(f"{table}.{column.name} does not exist")
        elif actual.type != column.postgres:
            found.append(f"{table}.{column.name} is {actual.type}, the model needs {column.postgres}")
        elif column.nullable and not actual.nullable:
            found.append(f"{table}.{column.name} is NOT NULL, but the model allows None")
        elif not actual.writable and column.name not in matched:
            found.append(f"{table}.{column.name} is generated, so the model cannot fill it")
    versioned = target.writer if isinstance(target.writer, Versioned) else None
    snapshot = isinstance(target.writer, Snapshot) and versioned is None
    writer_set = {versioned.valid_from} if versioned is not None else {"removed_at"} if snapshot else set()
    if not update:  # an Update never inserts
        for actual in info.columns.values():
            unfilled = not actual.nullable and not actual.has_default and actual.name not in names
            if unfilled and actual.name not in writer_set:
                found.append(f"{table}.{actual.name} is NOT NULL with no default, but the model does not fill it")
    key = row_key(target)
    if versioned is not None:
        found.extend(_version_mismatches(target, versioned, info))
    elif key is not None and frozenset(key) not in info.unique_keys:
        found.append(f"{table} has no unique key on ({', '.join(key)})")
    if snapshot:
        removed = info.columns.get("removed_at")
        if removed is None or removed.type != _TIMESTAMPTZ or not removed.nullable:
            found.append(f"{table}.removed_at must be a nullable {_TIMESTAMPTZ} column")
        elif not removed.writable:
            found.append(f"{table}.removed_at is generated, so the Snapshot writer cannot set it")
    return found
