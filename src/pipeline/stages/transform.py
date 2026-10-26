"""Turn accepted records into one polars frame per target table.

Public API:
    Transformed: Frozen result of ``transform``: pairs of a target and its frame.
    transform: Build the frames for a sequence of accepted records of one model.
    frame_of: Check a target's rows against its columns and build its frame.

"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import MAX_EMAX, MAX_PREC, MIN_EMIN, Context, Decimal
from typing import Final
from uuid import UUID

import polars as pl
from pydantic import BaseModel

from pipeline.checks.record import Record
from pipeline.checks.targets import Column, Target, target_of

#: Wide enough that moving a decimal point never rounds.
#: Decimal context with the largest precision and exponent range, so ``scaleb`` in ``_fits`` never rounds.
_EXACT: Final = Context(prec=MAX_PREC, Emax=MAX_EMAX, Emin=MIN_EMIN)


@dataclass(frozen=True, slots=True, kw_only=True)
class Transformed:
    """Frozen, keyword-only result of ``transform``: each target paired with its frame.

    Attributes:
        frames (tuple[tuple[Target, pl.DataFrame], ...]): One (target, frame) pair per target table, in load order:
            the root first, each parent before its children. Every target has a pair, its frame empty when no record
            has rows for it. ``transform`` builds it from a generator. Not checked by ``__post_init__`` (there is
            none).

    """

    frames: tuple[tuple[Target, pl.DataFrame], ...]  # one per target table, in load order: parents first


def transform[R: Record](model: type[R], accepted: Sequence[R], source_id: UUID) -> Transformed:
    """Build the frames for the accepted records of one model.

    Preconditions:
        ``model`` declares a table whose writer is not ``ReplaceChildren`` (checked by ``target_of``).

    Args:
        model: The record model; its ``__target__`` is the root target.
        accepted: Records that passed the checks.
        source_id: The source the records belong to, written as text to the root's source column.

    Returns:
        ``Transformed`` holding one pair per target: the root and every child target.

    Raises:
        TypeError: From ``target_of``, when ``model`` declares no table or loads as child rows.
        ValueError: From ``frame_of``, when a value does not fit its column.

    """
    root = target_of(model)
    rows: dict[Target, list[dict[str, object]]] = {}
    pending = [root]
    while pending:  # every target gets a frame, in load order, even when no record has rows for it
        target = pending.pop()
        rows[target] = []
        pending.extend(child for _, child in reversed(target.children))
    for record in accepted:
        _rows(root, record, {root.source_column: str(source_id)} if root.source_column else {}, rows)
    return Transformed(frames=tuple((target, frame_of(target, found)) for target, found in rows.items()))


def _rows(
    target: Target, part: BaseModel, parent: Mapping[str, object], out: Mapping[Target, list[dict[str, object]]]
) -> None:
    """Append the row for ``part`` to ``out[target]``, then the rows of its children.

    Each child row carries the key values in ``inherited``.

    Preconditions:
        ``out`` has a list for ``target`` and for every descendant target that would be reached.

    Args:
        target: The table the row is for.
        part: The record or child part to read.
        parent: Values passed down from the parent row (the source column for the root, inherited key values for a
            child).
        out: Rows collected so far, by target; lists are appended to in place.

    Returns:
        None.

    Raises:
        Nothing of its own. A column found in ``parent`` is copied and the path walk is skipped.

    """
    row: dict[str, object] = {}
    for column in target.columns:
        if column.name in parent:  # the source column of a record, or a key column a child inherits
            row[column.name] = parent[column.name]
            continue
        value: object = part
        for name in column.path:
            value = None if value is None else getattr(value, name)  # an absent optional part gives None
        row[column.name] = str(value) if column.postgres == "uuid" and value is not None else value
    out[target].append(row)
    inherited = {name: row[name] for _, child in target.children for name in child.inherited}
    for field, child in target.children:
        for item in getattr(part, field):
            _rows(child, item, inherited, out)


def frame_of(target: Target, rows: Sequence[Mapping[str, object]]) -> pl.DataFrame:
    """Check every row against the target's columns and build the target's frame from all rows.

    A value fits when it is ``None`` in a nullable column, or ``_fits`` accepts it.

    Preconditions:
        Every row has a key for every column of ``target``.

    Args:
        target: The table whose columns give the checks and the frame schema.
        rows: The rows to check and load, as produced by ``_rows``.

    Returns:
        A frame with one column per target column, in column order, typed by ``column.polars``; an empty frame with
        the target's schema when ``rows`` is empty (never ``None``).

    Raises:
        KeyError: When a row lacks a column's name.
        ValueError: When any value does not fit its column, naming the table, column and value (an ``int`` outside
            bigint's range included).

    """
    for row in rows:
        for column in target.columns:
            value = row[column.name]
            if not (column.nullable if value is None else _fits(column, value)):
                raise ValueError(
                    f"{target.table}.{column.name}: {value!r} does not fit "
                    f"{column.postgres}{'' if column.nullable else ' NOT NULL'}"
                )
    return pl.DataFrame(rows, schema={column.name: column.polars for column in target.columns}, strict=True)


def _fits(column: Column, value: object) -> bool:
    """Return whether a non-``None`` value can be stored in the column's polars type.

    ``bool`` is matched before ``int`` and ``datetime`` before ``date``, since each is a subclass of the next.

    Preconditions:
        ``column.polars`` of a Decimal column has a precision and scale: ``checks.targets`` always builds
        ``pl.Decimal(p, s)`` and refuses a Decimal without ``max_digits`` and ``decimal_places``.

    Args:
        column: The target column; only ``column.polars`` is read.
        value: The value to check.

    Returns:
        True for a ``bool`` in a Boolean column, an ``int`` from -2**63 to 2**63 - 1 in an Int64 column, any ``str``
        in a String column, an aware ``datetime`` in a Datetime column, a ``date`` in a Date column, or a finite
        ``Decimal`` in a Decimal column that it fills without rounding and within its precision. False for
        everything else.

    """
    # Polars would cast or round a value of another type, or null a decimal too wide, so each value is checked first
    dtype = column.polars
    match value:
        case bool():
            return isinstance(dtype, pl.Boolean)
        case int():
            return isinstance(dtype, pl.Int64) and -(2**63) <= value < 2**63  # bigint's range too
        case str():
            return isinstance(dtype, pl.String)
        case datetime():
            return isinstance(dtype, pl.Datetime) and value.utcoffset() is not None
        case date():
            return isinstance(dtype, pl.Date)
        case Decimal() if isinstance(dtype, pl.Decimal) and value.is_finite():
            scaled = value.scaleb(dtype.scale, _EXACT)
            return scaled == scaled.to_integral_value() and abs(scaled) < 10**dtype.precision
    return False
