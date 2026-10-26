"""Describe where a record model loads: its table, writer and columns, built from its field marks and type mapping.

Public API:
    To: Field mark that loads a single value into the named column.
    Rows: Field mark that loads a ``list`` of child parts as rows of the child's own table.
    Upsert: Writer that inserts rows or updates them on a conflict over ``key``.
    ReplaceChildren: Writer for a child table whose rows are replaced under each parent's key.
    ReplacePartition: Writer that replaces the rows whose ``column`` lies in the task's time frame.
    Snapshot: Writer keyed by ``key`` that sets ``removed_at`` itself.
    Update: Writer that updates existing rows matched on ``key``; ``missing`` says what happens to rows not returned.
    Writer: The union of the five writer classes.
    Column: One target column: its name, frame and Postgres types, nullability and path on the record.
    Target: A table, its writer, source column, columns and child targets.
    target_of: Return a model's top-level target.
    build_target: Build and check a model's target from its table, writer, source column and field marks.
    mismatches: List the ways a table in the catalog does not fit a target.

Private units:
    _columns: Collect the columns of a model's marked fields and computed fields, nested parts included.
    _unwrap: Strip type aliases, ``Annotated`` and ``Optional`` from an annotation.
    _mark: Return the one ``To`` or ``Rows`` in a field's metadata, or None.
    _child: Return the target of a ``Rows`` field's child part.
    _column: Build the ``Column`` for a ``To`` field.
    _column_type: Map an annotation to its frame type and Postgres type.
    _check_names: Check that no column name appears more than once.
    _check_writer: Check that a writer fits the columns.
    _key: Return a writer's key, or None for a writer without one.

"""

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

from pipeline.db.catalog import TableInfo

#: A ``schema.table`` name: two lower-case identifiers of ASCII letters, digits and ``_``, not starting with a digit,
#: joined by one dot. Used with ``fullmatch``.
_TABLE: Final = re.compile(r"[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*")
#: A bare column name: one lower-case identifier of ASCII letters, digits and ``_``, not starting with a digit. Used
#: with ``fullmatch``.
_COLUMN: Final = re.compile(r"[a-z_][a-z0-9_]*")

#: The Postgres type name of a ``timestamptz`` column, as ``format_type`` prints it.
_TIMESTAMPTZ: Final = "timestamp with time zone"
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
#: The Postgres types an ``Update`` with ``missing="zero"`` can set to zero: ``bigint`` or ``numeric(p,s)``. Used with
#: ``fullmatch``.
_NUMERIC: Final = re.compile(r"bigint|numeric\(\d+,\d+\)")


@dataclass(frozen=True, slots=True)
class To:
    """Field mark that loads a single value into the named column.

    ``__post_init__`` is not defined, so ``column`` is not validated here; ``_column`` checks it against ``_COLUMN``
    when the model's target is built.

    Attributes:
        column: The target column name.

    """

    column: str


@dataclass(frozen=True, slots=True)
class Rows:
    """Field mark that loads a ``list`` of child parts as rows of the child's own table.

    Only a ``list[SomePart]`` field of the class that declares the table may carry it, and ``SomePart`` must declare a
    table with ``ReplaceChildren()``; ``_columns`` and ``_child`` check this.

    """

    ...


@dataclass(frozen=True, slots=True)
class Upsert:
    """Writer that inserts rows or updates them on a conflict over ``key``.

    ``__post_init__`` is not defined, so ``key`` is not validated here; ``_check_writer`` checks it when the target is
    built.

    Attributes:
        key: Target column names: non-empty, distinct, not nullable, and including the source column when the table has
            one.

    """

    key: tuple[str, ...]  # target columns, the source column included when the table has one


@dataclass(frozen=True, slots=True)
class ReplaceChildren:
    """Writer for a child table whose rows are replaced under each parent's key.

    It has no key of its own: the child target inherits the parent writer's key columns. A target with this writer
    has no source column (``build_target`` refuses one) and is reached only through its parent's ``children``.

    """

    ...  # key = the columns inherited from the parent


@dataclass(frozen=True, slots=True)
class ReplacePartition:
    """Writer that replaces the rows whose ``column`` lies in the task's time frame.

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
    loads into ``removed_at`` when the target is built.

    Attributes:
        key: Target column names: non-empty, distinct, not nullable, and including the source column when the table has
            one.

    """

    key: tuple[str, ...]  # target columns, the source column included when the table has one


@dataclass(frozen=True, slots=True)
class Update:
    """Writer that updates existing rows matched on ``key``; ``missing`` says what happens to rows not returned.

    ``__post_init__`` is not defined, so nothing is validated here; ``_check_writer`` checks, when the target is built,
    ``key``, that some column lies outside it, and for ``missing="zero"`` that every column outside it is numeric.

    Attributes:
        key: Target column names: non-empty, distinct, not nullable, and including the source column when the table has
            one.
        missing: ``"keep"`` leaves the source's rows the connector did not return; ``"zero"`` sets their columns outside
            ``key`` to zero.

    """

    key: tuple[str, ...]  # target columns, the source column included when the table has one
    missing: Literal["keep", "zero"] = "keep"  # what happens to the source's rows the connector did not return


#: The writer of a target: how its frame is written to its table.
type Writer = Upsert | ReplaceChildren | ReplacePartition | Snapshot | Update


@dataclass(frozen=True, slots=True)
class Column:
    """One target column: its name, frame and Postgres types, nullability and path on the record.

    ``__post_init__`` is not defined, so nothing is validated here; ``build_target`` makes every instance.

    Attributes:
        name: The target column name, matching ``_COLUMN``.
        polars: The column's type in the frame.
        postgres: The column's type exactly as ``format_type`` prints it.
        nullable: True when the field allows None, or lies inside a part that does.
        path: Field names from the record to the value, nested parts included; ``()`` for the source column. In a child
            target, an inherited column's path is on the parent record and the child's own columns' paths are on the
            child part.

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
    """A table, its writer, source column, columns and child targets.

    ``__post_init__`` is not defined, so nothing is validated here; every instance comes from ``build_target``, which
    checks the names, the writer against the columns, and that column names are distinct.

    Attributes:
        table: The table as ``schema.table``, matching ``_TABLE``.
        writer: How the frame is written to ``table``.
        source_column: The column the task's source id is written to, or None; always None for ``ReplaceChildren``.
        columns: Every column the frame has, in frame order: the source column first, then inherited, then own.
        inherited: The column names copied from the parent row (the parent writer's key); ``()`` for a top-level
            target.
        children: ``(Rows field name, child target)`` pairs in field order; a child target has no children.

    """

    table: str  # "schema.table"
    writer: Writer
    source_column: str | None
    columns: tuple[
        Column, ...
    ]  # every column the frame has, in frame order: source column first, then inherited, then own
    inherited: tuple[str, ...]  # columns copied from the parent row; () for a top-level target
    children: tuple[tuple[str, Target], ...]  # (the Rows field name, the child's target), in field order


def target_of(model: type[BaseModel]) -> Target:
    """Return a model's top-level target.

    Preconditions:
        None.

    Args:
        model: The record model; its ``__target__`` attribute is read, and a missing attribute counts as None.

    Returns:
        ``model.__target__``.

    Raises:
        TypeError: ``model`` declares no table, or its writer is ``ReplaceChildren`` (its full target is in its
            parent's ``children``).

    """
    target: Target | None = getattr(model, "__target__", None)
    if target is None:
        raise TypeError(f"{model.__name__} declares no table")
    if isinstance(target.writer, ReplaceChildren):
        raise TypeError(f"{model.__name__} loads as child rows; its target is in its parent's children")
    return target


def build_target(model: type[BaseModel], table: str, writer: Writer, source_column: str | None) -> Target:
    """Build and check a model's target from its table, writer, source column and field marks.

    The columns are the source column (``uuid``, not nullable, path ``()``) when given, then the columns ``_columns``
    collects. Each ``Rows`` field gives a child whose target is rebuilt with the writer key's columns (inherited)
    before its own and ``inherited`` set to the key names. A child cannot have children of its own: its writer is
    ``ReplaceChildren``, which has no key, so its own build refused any ``Rows``. The model's table and its children's
    tables must all differ.

    Preconditions:
        None.

    Args:
        model: The model declaring the table.
        table: The table as ``schema.table`` in lower case.
        writer: How rows are written to ``table``.
        source_column: The column the task's source id is written to, or None.

    Returns:
        The target, with ``inherited=()`` and one child per ``Rows`` field in field order.

    Raises:
        ValueError: ``table`` or ``source_column`` is not a lower-case name of the right form; from ``_columns``, for a
            malformed ``To`` column; from ``_check_writer``, for a writer that does not fit the columns.
        TypeError: ``source_column`` is given with ``ReplaceChildren``; ``model`` is not complete; ``Rows`` is used with
            a writer that has no key; two targets load into one table; from ``_columns``, for a misused mark, a type
            with no column type or a part that contains itself; from ``_check_names``, for a repeated column name in
            the model or in a child.

    """
    name = model.__name__
    if not _TABLE.fullmatch(table):
        raise ValueError(f"{name}: table must be schema.table in lower case, got {table!r}")
    if source_column is not None and not _COLUMN.fullmatch(source_column):
        raise ValueError(f"{name}: source_column must be a lower-case name, got {source_column!r}")
    if source_column is not None and isinstance(writer, ReplaceChildren):
        raise TypeError(f"{name}: a ReplaceChildren table has no source_column; the source comes with the parent's key")
    if not model.__pydantic_complete__:
        raise TypeError(f"{name} declares a table, so its field types must be defined first")
    source = (Column(source_column, pl.String(), "uuid", False, ()),) if source_column is not None else ()
    rows: list[tuple[str, Target]] = []
    columns = (*source, *_columns(model, (), False, rows, (model,)))
    _check_names(name, columns)
    _check_writer(name, writer, columns, source_column)
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
        children.append((field, Target(child.table, child.writer, child.source_column, child_columns, key or (), ())))
    return Target(table, writer, source_column, columns, (), tuple(children))


def _columns(
    model: type[BaseModel],
    path: tuple[str, ...],
    optional: bool,
    rows: list[tuple[str, Target]] | None,
    inside: tuple[type[BaseModel], ...],  # the models this walk is in, so a part that contains itself is refused
) -> list[Column]:
    """Collect the columns of a model's marked fields and computed fields, nested parts included.

    Fields come first, in field order, then computed fields. A ``To`` field gives one column. An unmarked field whose
    type is a part gives the part's columns in its place; a computed field that returns a part does not. A ``Rows``
    field appends ``(field name, child target)`` to ``rows`` and gives no column. Other unmarked fields, unmarked
    lists included, are skipped.

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
        TypeError: ``To`` on a list; ``Rows`` on a nested part's field, a non-list or a computed field; a mark on a
            part field; a part field whose part has its own table; a part field whose part contains itself (its
            part is in ``inside``); from ``_mark``, ``_child`` and ``_column``.
        ValueError: From ``_column``, for a malformed ``To`` column.

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
            if mark is not None:
                raise TypeError(f"{where}: {type(mark).__name__} is not for a part; mark the part's own fields")
            if getattr(annotation, "__target__", None) is not None:
                raise TypeError(
                    f"{where}: {annotation.__name__} has its own table, so only a list of it loads, with Rows"
                )
            if annotation in inside:
                raise TypeError(
                    f"{where}: {annotation.__name__} contains itself, so it cannot be flattened into columns"
                )
            columns.extend(_columns(annotation, here, optional or nullable, None, (*inside, annotation)))
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
    """Strip type aliases, ``Annotated`` and ``Optional`` from an annotation.

    Layers are stripped from the outside in until none is left. Each ``Annotated`` adds its extras to the metadata; a
    ``FieldInfo`` extra adds its ``metadata`` items rather than itself. ``X | None`` becomes ``X`` and sets nullable.
    A union of two or more types besides None is returned whole, still holding None, with nullable set.

    Preconditions:
        None.

    Args:
        annotation: The annotation to strip.
        metadata: Metadata already found, placed first in the result.

    Returns:
        The stripped annotation, ``metadata`` followed by the metadata found, and whether None is allowed.

    Raises:
        Nothing.

    """
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
    """Return the target of a ``Rows`` field's child part.

    Preconditions:
        ``annotation`` is ``list[X]`` with one argument (``_columns`` calls this only when ``get_origin`` is
        ``list``).

    Args:
        where: ``Model.field``, used in the error message.
        annotation: The field's stripped annotation.
        nullable: Whether the field allows None.

    Returns:
        ``X.__target__``, whose writer is ``ReplaceChildren``.

    Raises:
        TypeError: The field allows None, ``X`` declares no table, or ``X``'s writer is not ``ReplaceChildren``.

    """
    (item,) = get_args(annotation)
    target: Target | None = getattr(item, "__target__", None)
    if nullable or target is None or not isinstance(target.writer, ReplaceChildren):
        raise TypeError(f"{where}: Rows needs list[SomePart], where SomePart declares a table with ReplaceChildren()")
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
        ValueError: ``mark.column`` does not match ``_COLUMN``.
        TypeError: From ``_column_type``, for a type with no column type.

    """
    if not _COLUMN.fullmatch(mark.column):
        raise ValueError(f"{where}: To column must be a lower-case name, got {mark.column!r}")
    polars, postgres = _column_type(where, annotation, metadata)
    return Column(mark.column, polars, postgres, nullable, path)


def _column_type(where: str, annotation: Any, metadata: Sequence[object]) -> tuple[pl.DataType, str]:
    """Map an annotation to its frame type and Postgres type.

    ``Decimal`` maps to ``pl.Decimal(p, s)`` and ``numeric(p,s)``, with ``p`` and ``s`` the last non-None
    ``max_digits`` and ``decimal_places`` found on the metadata items. Any other annotation is looked up in
    ``_SCALARS``.

    Preconditions:
        None.

    Args:
        where: ``Model.field``, used in error messages.
        annotation: The stripped annotation.
        metadata: The field's metadata.

    Returns:
        The frame type and the Postgres type as ``format_type`` prints it.

    Raises:
        TypeError: A ``Decimal`` lacks ``max_digits`` or ``decimal_places``, or the annotation is not in ``_SCALARS``
            (an unhashable annotation included).

    """
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


def _check_writer(name: str, writer: Writer, columns: Sequence[Column], source_column: str | None) -> None:
    """Check that a writer fits the columns.

    A writer key must be non-empty, name distinct target columns that do not allow None, and include the source column
    when there is one. ``ReplacePartition`` needs its column to be a target column of type ``timestamptz`` or
    ``date`` that does not allow None. ``Snapshot`` refuses a column named ``removed_at``. ``Update`` needs a column
    outside its key, and with ``missing="zero"`` every such column must be ``bigint`` or ``numeric(p,s)``.
    ``ReplaceChildren`` and ``Upsert`` have no check beyond the key's.

    Preconditions:
        Column names in ``columns`` are distinct (``build_target`` calls ``_check_names`` first).

    Args:
        name: The model name, used in error messages.
        writer: The writer to check.
        columns: The target's columns.
        source_column: The target's source column, or None.

    Returns:
        None.

    Raises:
        ValueError: Any of the checks above fails.

    """
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
        if source_column is not None and source_column not in key:
            raise ValueError(f"{name}: the writer's key must include the source column {source_column!r}")
    match writer:
        case ReplacePartition(column=column):
            if column not in by_name or by_name[column].postgres not in (_TIMESTAMPTZ, "date"):
                raise ValueError(f"{name}: ReplacePartition needs a target column of type {_TIMESTAMPTZ} or date")
            if by_name[column].nullable:
                raise ValueError(f"{name}: the ReplacePartition column {column!r} cannot allow None")
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


def mismatches(target: Target, info: TableInfo | None) -> list[str]:
    """List the ways a table in the catalog does not fit a target.

    Each target column gets at most one message, the first that applies: it does not exist, its type differs from
    ``Column.postgres`` (compared as exact strings), it is ``NOT NULL`` while the model allows None, or it is
    not writable (a generated or ``GENERATED ALWAYS`` identity column). Then each table column that is ``NOT NULL``
    with no default and not a target column gets a message, except ``removed_at`` for a ``Snapshot``. A writer key
    needs a unique key on exactly its columns, and a ``Snapshot`` needs a nullable ``timestamptz`` ``removed_at``.

    Preconditions:
        ``info`` is ``table_info``'s result for ``target.table``.

    Args:
        target: The target to check.
        info: The table as the catalog describes it, or None when ``table_info`` found no table.

    Returns:
        One message per mismatch, in the order above; ``[]`` when the table fits. A single message when ``info`` is
        None.

    Raises:
        Nothing.

    """
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
