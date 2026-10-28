"""Turn accepted records into one polars frame per target table.

Public API:
    Transformed: Frozen result of ``transform``: pairs of a target and its frame.
    transform: Build the frames for a sequence of accepted records of one model.
    frame_of: Check a target's rows against its columns and build its frame.

"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import MAX_EMAX, MAX_PREC, MIN_EMIN, Context, Decimal
from typing import Final
from uuid import UUID

import polars as pl
from pydantic import BaseModel

from pipeline.checks.record import Keyed, Record
from pipeline.checks.targets import Column, ReplaceChildren, Target, target_of
from pipeline.db.pool import storable_text

#: Decimal context with the largest precision and exponent range, so ``scaleb`` in ``_fits`` and ``quantize`` in
#: ``frame_of`` never round. Its traps are the defaults, so a result past ``MAX_EMAX`` raises ``decimal.Overflow``
#: rather than rounding; ``_fits`` treats that as a value that does not fit.
_EXACT: Final = Context(prec=MAX_PREC, Emax=MAX_EMAX, Emin=MIN_EMIN)

#: The first and last instants Python's ``datetime`` can hold in UTC, as ``frame.iter_rows()`` in
#: ``writers.staging.copy_in`` converts each value back to one; ``_fits`` refuses an aware ``datetime`` outside them.
_FIRST_UTC: Final = datetime.min.replace(tzinfo=UTC)
_LAST_UTC: Final = datetime.max.replace(tzinfo=UTC)

#: Text forms of a uuid that Postgres's ``uuid`` input accepts, in ASCII hex of either case: canonical 8-4-4-4-12 or
#: 32 digits, either one optionally in braces.
_UUID_TEXT: Final = re.compile(
    r"(?:\{(?:[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}|[0-9a-fA-F]{32})\}"
    r"|[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}|[0-9a-fA-F]{32})"
)


@dataclass(frozen=True, slots=True, kw_only=True)
class Transformed:
    """Frozen, keyword-only result of ``transform``: each target paired with its frame.

    ``__post_init__`` refuses a target that appears twice, a child target that comes before its parent or whose
    parent is absent, and a frame whose columns differ from its target's; ``load`` relies on all of these. Otherwise
    completeness (a pair for every target) is not checked; ``transform`` gives every target one.

    Attributes:
        frames (tuple[tuple[Target, pl.DataFrame], ...]): (target, frame) pairs, no target twice, every child's
            parent present and before it, and every frame's column names, order and types those of its target's
            ``columns`` (all checked). From ``transform``: one pair per target, the root first, then its children in
            field order, a frame empty when no record has rows for it.

    """

    frames: tuple[tuple[Target, pl.DataFrame], ...]  # one per target table, in load order: parents first

    def __post_init__(self) -> None:
        """Refuse frames in which a target appears twice, a child target is misplaced, or a frame misfits its target.

        A child target is one whose writer is ``ReplaceChildren``; its parent is the target listing it in
        ``children``.

        Preconditions:
            None.

        Returns:
            None.

        Raises:
            ValueError: A target appears more than once; a target in some target's ``children`` comes earlier in
                ``frames`` than that target; a child target is in no other target's ``children`` in ``frames``; or a
                frame's column names, order or types differ from its target's ``columns``. The message names the
                tables.

        """
        seen: set[Target] = set()
        adopted: set[Target] = set()  # children of the targets seen so far
        for target, frame in self.frames:
            if target in seen:
                raise ValueError(f"Transformed: {target.table} appears more than once")
            early = [child.table for _, child in target.children if child in seen]
            if early:
                raise ValueError(f"Transformed: {', '.join(early)} comes before its parent {target.table}")
            expected = [(column.name, column.polars) for column in target.columns]
            if list(frame.schema.items()) != expected:
                raise ValueError(f"Transformed: the frame for {target.table} does not have its target's columns")
            seen.add(target)
            adopted.update(child for _, child in target.children)
        orphans = [
            target.table for target in seen if isinstance(target.writer, ReplaceChildren) and target not in adopted
        ]
        if orphans:
            raise ValueError(f"Transformed: {', '.join(sorted(orphans))} has no parent in frames")


def transform[R: Record](model: type[R], accepted: Sequence[Keyed[R]], source_id: UUID) -> Transformed:
    """Build the frames for the accepted records of one model.

    Every target gets an empty row list first, in load order (the root, then its children in field order; a child
    has no children, per ``build_target``). Each record then adds its row and its child rows through ``_rows``, and
    each list is turned into a frame by ``frame_of``. The root's key columns (the columns of the ``record_key`` fields)
    are written from the record's computed key (``Keyed.key``, the key Validate de-duplicated on), never from its
    attributes, a ``uuid`` value as text; children inherit them from the root row, so their parent-key columns hold the
    same values. Every other column comes from the parsed record. A record that does not fit stops the whole call; no
    frame is returned for the others.

    Preconditions:
        Every record in ``accepted`` is an instance of ``model`` and its ``key`` is ``key_of`` of ``model`` and its
        payload: the job runner passes ``Validated.accepted`` with the model of the same validation. This is not
        checked.

    Args:
        model: The record model; its ``__target__`` is the root target.
        accepted: Records that passed the checks, each with its key; may be empty, giving empty frames.
        source_id: The source the records belong to, written as text to the root's source column when it has one.

    Returns:
        ``Transformed`` holding one pair per target, the root first, then every child target in field order.

    Raises:
        TypeError: From ``target_of``, when ``model`` declares no table or loads as child rows.
        ValueError: From ``frame_of``, when a value does not fit its column.
        AttributeError: From ``_rows``, when a record is not an instance of ``model`` and lacks a column's field.
        IndexError: When a record's ``key`` is shorter than ``model.record_key`` (the precondition broken).
        Exception: Whatever a computed field's property raises when ``_rows`` reads it; the checks never evaluate
            computed fields, so this is the first place they run.

    """
    root = target_of(model)
    # every target gets a frame, in load order, even when no record has rows for it; a child has no children
    rows: dict[Target, list[dict[str, object]]] = {root: [], **{child: [] for _, child in root.children}}
    source = {root.source_column: str(source_id)} if root.source_column else {}
    key_columns = [  # each key column with its field's position in record_key
        (column, model.record_key.index(column.path[0]))
        for column in root.columns
        if len(column.path) == 1 and column.path[0] in model.record_key
    ]
    for keyed in accepted:
        key = {
            column.name: str(keyed.key[index]) if column.postgres == "uuid" else keyed.key[index]
            for column, index in key_columns
        }
        _rows(root, keyed.record, source | key, rows)
    return Transformed(frames=tuple((target, frame_of(target, found)) for target, found in rows.items()))


def _rows(
    target: Target, part: BaseModel, parent: Mapping[str, object], out: Mapping[Target, list[dict[str, object]]]
) -> None:
    """Append the row for ``part`` to ``out[target]``, then the rows of its children.

    A column named in ``parent`` takes its value from there, as given; any other column follows its path of attributes
    from ``part``, giving None once an optional part on the way is None. A computed field's property runs when its
    column is read. A ``uuid`` column's value is turned into text. Each child row then gets the parent row's values for
    the child's ``inherited`` columns (the parent writer's key, which ``build_target`` makes from this target's columns,
    so every name is in ``row``). A ``Rows`` field is never None: ``checks.targets._child`` refuses a nullable one.

    Preconditions:
        ``out`` has a list for ``target`` and for every child target (``transform`` makes one for each). ``part`` is
        an instance of the model ``target`` was built from, so every column path and ``Rows`` field resolves.

    Args:
        target: The table the row is for.
        part: The record or child part to read.
        parent: Values passed down: the source column and the key columns (from the record's computed key) for the
            root, the inherited key values for a child.
        out: Rows collected so far, by target; lists are appended to in place.

    Returns:
        None.

    Raises:
        Exception: Whatever a computed field's property raises. Nothing else when the preconditions hold;
            ``AttributeError`` when ``part`` is not an instance of ``target``'s model.

    """
    row: dict[str, object] = {}
    for column in target.columns:
        if column.name in parent:  # the source or a key column of a record, or a key column a child inherits
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

    A value fits when it is ``None`` in a nullable column, or ``_fits`` accepts it. Keys of a row that are not column
    names are ignored. An aware ``datetime`` is converted to UTC by polars, as every Datetime column ``checks.targets``
    builds is ``pl.Datetime("us", "UTC")``. Each ``Decimal`` is rescaled to its column's scale with ``quantize`` in
    ``_EXACT`` before the frame is built, so polars gets at most the column's precision in digits whatever the
    value's own digits and exponent (``Decimal("0E+39")`` and ``Decimal("0E-45")`` load into ``numeric(10,2)`` as
    ``0.00``); the rescaling is exact, as ``_fits`` accepted the value.

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
            bigint's range and a ``Decimal`` whose exponent is near ``MAX_EMAX`` included).

    """
    built: list[dict[str, object]] = []
    for row in rows:
        fitted: dict[str, object] = {}
        for column in target.columns:
            value = row[column.name]
            if not (column.nullable if value is None else _fits(column, value)):
                raise ValueError(
                    f"{target.table}.{column.name}: {value!r} does not fit "
                    f"{column.postgres}{'' if column.nullable else ' NOT NULL'}"
                )
            if isinstance(value, Decimal) and isinstance(column.polars, pl.Decimal):  # exact: _fits accepted it
                value = value.quantize(Decimal(1).scaleb(-column.polars.scale), context=_EXACT)
            fitted[column.name] = value
        built.append(fitted)
    return pl.DataFrame(built, schema={column.name: column.polars for column in target.columns}, strict=True)


def _is_uuid(text: str) -> bool:
    """Return whether ``text`` is one of the uuid text forms ``_UUID_TEXT`` allows.

    Preconditions:
        None.

    Args:
        text: The text to check.

    Returns:
        True when ``_UUID_TEXT`` matches the whole of ``text``; False otherwise, so ``urn:uuid:...`` and non-ASCII
        digits, which Python's ``UUID`` accepts and Postgres refuses, are refused.

    Raises:
        Nothing.

    """
    return _UUID_TEXT.fullmatch(text) is not None


def _fits(column: Column, value: object) -> bool:
    """Return whether a non-``None`` value can be stored in the column's polars type.

    ``bool`` is matched before ``int`` and ``datetime`` before ``date``, since each is a subclass of the next. A
    ``Decimal`` is scaled by the column's scale in ``_EXACT``, and its magnitude is then taken with ``copy_abs``, which
    is exact and never rounds or traps.

    Preconditions:
        ``column.polars`` of a Decimal column has a precision and scale: ``checks.targets`` always builds
        ``pl.Decimal(p, s)`` and refuses a Decimal without ``max_digits`` and ``decimal_places``.

    Args:
        column: The target column; ``column.polars`` is read, and ``column.postgres`` for a ``str``.
        value: The value to check.

    Returns:
        True when the value's type matches the column's polars type and the value is in range:

        - a ``bool`` in a Boolean column;
        - an ``int`` from -2**63 to 2**63 - 1 in an Int64 column;
        - a ``str`` Postgres ``text`` can store (no NUL and no surrogate, the rule of ``db.pool.storable_text``) in a
          String column, and, when the column is ``uuid`` (its frame type is String), also a uuid text form
          (``_is_uuid``);
        - an aware ``datetime`` whose UTC instant lies in years 1 to 9999 in a Datetime column;
        - a ``date`` in a Date column;
        - a finite ``Decimal`` in a Decimal column that it fills without rounding and whose exact magnitude, scaled
          by the column's scale, is below ``10**precision``.

        False for everything else, including a ``Decimal`` whose scaling or integral check raises an
        ``ArithmeticError`` (``decimal.Overflow`` when its exponent plus the column's scale passes ``MAX_EMAX``), an
        aware ``0001-01-01T00:00+05:00`` (year 0 in UTC), and a ``str`` holding a NUL character or ``"abc"`` in a
        ``uuid`` column, so these are refused here rather than failing at load.

    Raises:
        Nothing.

    """
    # Polars would cast or round a value of another type, or null a decimal too wide, so each value is checked first
    dtype = column.polars
    match value:
        case bool():
            return isinstance(dtype, pl.Boolean)
        case int():
            return isinstance(dtype, pl.Int64) and -(2**63) <= value < 2**63  # bigint's range too
        case str():
            storable = isinstance(dtype, pl.String) and storable_text(value) is value  # the same string when storable
            return storable and (column.postgres != "uuid" or _is_uuid(value))
        case datetime():
            aware = value.utcoffset() is not None
            return isinstance(dtype, pl.Datetime) and aware and _FIRST_UTC <= value <= _LAST_UTC
        case date():
            return isinstance(dtype, pl.Date)
        case Decimal() if isinstance(dtype, pl.Decimal) and value.is_finite():
            try:
                scaled = value.scaleb(dtype.scale, _EXACT)
                return scaled == scaled.to_integral_value() and scaled.copy_abs() < 10**dtype.precision
            except ArithmeticError:  # decimal.Overflow: far too large for any column
                return False
    return False
