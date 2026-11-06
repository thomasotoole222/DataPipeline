"""Base models for parsed connector records, the check that each record type declares a usable key, and key lookup.

Public API:
    BOOL_VALUES: The only values a boolean field accepts, each mapped to its ``bool``.
    Bool: A boolean field's type: its input mapped through ``BOOL_VALUES``, then checked as a strict ``bool``.
    Int32: An ``int`` field bounded to -2**31 to 2**31 - 1 and marked ``Width(32)``, so it loads into ``integer``.
    Int16: An ``int`` field bounded to -2**15 to 2**15 - 1 and marked ``Width(16)``, so it loads into ``smallint``.
    DateKey: An ``int`` field read from a ``YYYY-MM-DD`` date or a ``YYYYMMDD`` int, held as the ``YYYYMMDD`` int.
    TaskDay: Field mark that fills a top-level field of a Record with the task's day as a ``YYYYMMDD`` int.
    task_day_fields: Return the names of a model's fields marked ``TaskDay``, in field order.
    Part: Frozen pydantic model that keeps unknown fields; the base for nested parts of a record.
    Record: A ``Part`` whose subclasses must declare ``record_key`` and ``record_version``, checked at class creation;
        each key field must be required, have a type defined when the model is declared, not accept ``None`` and have
        a type and metadata allowed by ``_key_refusal``; a subclass may not set ``strict=True`` in its
        ``model_config``; it may declare a ``reshape`` applied to each landed record before it is parsed.
    Keyed: A parsed record together with its key from ``key_of``.
    key_of: Return a record's key values in ``record_key`` order, read from its landed payload.

A model may declare its target table, writer and source column as class keywords
(``class Order(Record, table="sales.order", writer=Upsert(...))``); the target is built by ``build_target``
(checks.targets) when the class is defined. A Record may instead declare ``container=True``: it has no table and
fills only its ``Rows`` children's tables, and its target is built by ``build_container`` (checks.targets).

Private units:
    _to_bool: Return the ``bool`` that ``BOOL_VALUES`` maps a boolean field's input to, or the input unchanged.
    _DateKeyInput: The mark ``DateKey`` carries, which reads a date key's input before its ``int`` checks.
    _date_key: Return a date key's input as its ``YYYYMMDD`` int.
    _annotated: Return the field's type annotated with its own metadata, as a key field's form is checked.
    _allows_none: Return whether a field accepts ``None``.
    _key_refusal: Return why a type form is refused for a key field, or ``None`` when it is allowed.
    _declarative: Return whether a metadata item is a declarative constraint or mark (``To``, ``Width``,
        ``TaskDay``, ``DateKey``'s), which a key field may carry.
    _check_task_days: Check that every ``TaskDay`` field of a model is a Record's plain ``int`` field with no alias.
    _key_model: Return a model of only a Record's ``record_key`` fields, built once per record model.
    _check_table_key: Check that a Record's writer key is its source column plus its ``record_key`` columns.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import IntEnum, StrEnum
from functools import cache
from types import MappingProxyType
from typing import Annotated, Any, ClassVar, Final, Literal, NewType, TypeAliasType, get_args, get_origin
from uuid import UUID

from annotated_types import BaseMetadata, Ge, GroupedMetadata, Le, Predicate
from pydantic import (
    AllowInfNan,
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    GetCoreSchemaHandler,
    PydanticSchemaGenerationError,
    PydanticUserError,
    Strict,
    TypeAdapter,
    ValidationError,
    create_model,
)
from pydantic.fields import FieldInfo
from pydantic_core import core_schema

from pipeline.checks.targets import (
    Snapshot,
    Target,
    To,
    Update,
    Upsert,
    Width,
    Writer,
    alias_value,
    build_container,
    build_target,
)

#: The classes a key field's type may be, besides a ``StrEnum`` or ``IntEnum`` subclass; ``float`` is not one, nor a
#: plain ``datetime``, which accepts naive values: only Pydantic's ``AwareDatetime``, which refuses them.
_KEY_CLASSES: Final = (str, int, bool, date, AwareDatetime, Decimal, UUID)

#: The only values a boolean field accepts. Booleans are mapped here, in Validate, before the model's strict bool
#: check, and not cast in Transform, because Polars cannot cast text to Boolean: Series.cast(pl.Boolean) on text
#: raises InvalidOperationError ("casting from Utf8View to Boolean not supported", strict or not), building a strict
#: frame fails the whole batch on one text value, building a non-strict frame silently turns it into null, and an
#: integer cast turns any non-zero value (2) into True. Mapping in Validate is allowed only because it is a constant
#: O(1) hash-map lookup over primitive keys (bool and str) that needs no conversion logic beyond trimming and
#: lower-casing text. 1 and 0 hash and compare equal to True and False, so they are the same dict keys and match
#: those entries; floats and Decimals are never looked up. Any other value fails the strict bool check, so the
#: record is quarantined by the schema rule.
BOOL_VALUES: Final[Mapping[object, bool]] = MappingProxyType(
    {
        True: True,
        False: False,
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
        "t": True,
        "f": False,
        "on": True,
        "off": False,
    }
)


def _to_bool(value: object) -> object:
    """Return the ``bool`` that ``BOOL_VALUES`` maps a boolean field's input to, or the input unchanged.

    Runs before the field's strict ``bool`` check. Text is trimmed and lower-cased before the lookup; a ``bool`` or
    ``int`` is looked up as it is, so ``1`` and ``0`` find ``True`` and ``False``. Any other value, a ``float`` or
    ``Decimal`` included, is never looked up.

    Preconditions:
        None.

    Args:
        value: The field's input, as JSON decoded it.

    Returns:
        ``BOOL_VALUES``' ``bool`` for ``value``, or ``value`` unchanged when it is not a key, so the strict ``bool``
        check after it fails the record.

    Raises:
        None.

    """
    if isinstance(value, str):
        return BOOL_VALUES.get(value.strip().lower(), value)
    if isinstance(value, int):  # a bool is an int too; 1 and 0 find the keys True and False
        return BOOL_VALUES.get(value, value)
    return value  # a float, Decimal or anything else is never looked up, so the strict bool check refuses it


#: A boolean field: mapped through BOOL_VALUES, then checked strictly.
Bool = Annotated[bool, BeforeValidator(_to_bool), Strict()]

#: An int field loaded into an ``integer`` column (Postgres ``int4``), e.g. a ``YYYYMMDD`` date key. ``Ge`` and ``Le``
#: bound it to the column's range, so a value outside it fails model validation in Validate and the record is held
#: rather than the load failing; ``Width(32)`` makes ``_column_type`` (checks.targets) give the column ``integer``.
#: Both are declarative, so an Int32 may be a key field.
Int32 = Annotated[int, Ge(-(2**31)), Le(2**31 - 1), Width(32)]

#: An int field loaded into a ``smallint`` column (Postgres ``int2``), e.g. a coded status or an ``HHMM`` time key.
#: ``Ge`` and ``Le`` bound it to the column's range, so a value outside it fails model validation in Validate and the
#: record is held rather than the load failing; ``Width(16)`` makes ``_column_type`` (checks.targets) give the column
#: ``smallint``. Both are declarative, so an Int16 may be a key field.
Int16 = Annotated[int, Ge(-(2**15)), Le(2**15 - 1), Width(16)]


def _date_key(value: object) -> int:
    """Return a date key's input as its ``YYYYMMDD`` int.

    Preconditions:
        None.

    Args:
        value: The field's input, as JSON decoded it.

    Returns:
        ``year * 10000 + month * 100 + day`` for a ``str`` written ``YYYY-MM-DD``; an ``int`` (not a ``bool``) as it
        is, once its digits are found to name a real date.

    Raises:
        ValueError: ``value`` is a ``str`` that is not a date written ``YYYY-MM-DD`` (the compact ``"20261104"``
            included), an ``int`` whose ``YYYYMMDD`` digits name no date (``20260230``, ``0``, a negative one), or of
            any other type, a ``bool`` and a ``float`` included; Pydantic turns it into a ``ValidationError``.

    """
    if isinstance(value, str):
        day = date.fromisoformat(value)
        if day.isoformat() != value:
            raise ValueError(f"a date key is written YYYY-MM-DD, not {value!r}")
        return day.year * 10000 + day.month * 100 + day.day
    if isinstance(value, int) and not isinstance(value, bool):
        date(value // 10000, value // 100 % 100, value % 100)  # raises ValueError when the digits name no date
        return value
    raise ValueError(f"a date key is a YYYY-MM-DD date or a YYYYMMDD int, not a {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class _DateKeyInput:
    """The mark ``DateKey`` carries, which reads a date key's input before its ``int`` checks.

    A declarative mark in ``_declarative``'s sense, so a ``DateKey`` may be a key field: it has no fields, and its
    one piece of code, ``_date_key``, is fixed here. Placed last in ``DateKey``'s metadata, so its validator wraps
    the ``int`` schema and its bounds and runs first, on the raw input.

    """

    def __get_pydantic_core_schema__(self, source: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        """Return the field's schema with ``_date_key`` run on its input before it.

        Preconditions:
            None.

        Args:
            source: The annotated type, ``int``.
            handler: Builds the schema of ``source`` with the metadata before this mark.

        Returns:
            A before-validator schema of ``_date_key`` around the handler's schema.

        Raises:
            Exception: Whatever ``handler`` raises building the inner schema.

        """
        return core_schema.no_info_before_validator_function(_date_key, handler(source))


#: An int field read from a ``YYYY-MM-DD`` date (``"2026-11-04"``) or an int already in ``YYYYMMDD`` (``20261104``), and
#: held as the ``YYYYMMDD`` int, e.g. a report's ``stat_time_day`` as a ``date_key``. Any other input, and a date that
#: does not exist, fails model validation in Validate, so the record is held. ``Ge``, ``Le`` and ``Width(32)`` make it
#: load into ``integer``, as ``Int32`` does. Every item is declarative, so a DateKey may be a key field.
DateKey = Annotated[int, Ge(10101), Le(99991231), Width(32), _DateKeyInput()]


@dataclass(frozen=True, slots=True)
class TaskDay:
    """Field mark that fills a top-level field of a Record with the task's day as a ``YYYYMMDD`` int.

    Used like ``To``, and usually with it: ``date_key: Annotated[Int32, TaskDay(), To("date_key")]``. Validate writes
    the day of the task's ``"day"`` partition (its ``partition_key``, ``YYYY-MM-DD``) into each landed record's field
    of that name, over any value already there, before the record is validated (see ``stages.validate``), so the
    field's own constraints check it and it may be a ``record_key`` field. The mark has no fields and runs no code:
    Pydantic ignores it when it validates.

    Rules, checked when the model is declared by ``_check_task_days``: the field is a field of a ``Record``, not of a
    nested part (only a record's own payload is filled); its type is ``int`` (``Int32`` included), not optional; it has
    no alias, since the value is written under the field's name; and it carries one ``TaskDay``. ``BatchJob`` refuses
    a model with a ``TaskDay`` field unless the job's partition is ``"day"``.

    """


def task_day_fields(model: type[BaseModel]) -> tuple[str, ...]:
    """Return the names of a model's fields marked ``TaskDay``, in field order.

    Only the model's own fields are read, not those of a nested part.

    Preconditions:
        None.

    Args:
        model: The record model.

    Returns:
        The names; ``()`` when no field is marked.

    Raises:
        Nothing.

    """
    fields = model.model_fields.items()
    return tuple(name for name, field in fields if any(isinstance(item, TaskDay) for item in field.metadata))


class Part(BaseModel):
    """Frozen pydantic model that keeps unknown fields; the base for nested parts of a record.

    ``extra="allow"`` keeps input keys that are not declared fields as extra attributes instead of rejecting or
    dropping them. ``frozen=True`` makes assignment to fields raise and gives instances a ``__hash__`` over the
    declared field values; extra values are not hashed.

    A subclass may declare where it loads with three class keywords: ``table`` (``"schema.table"``), ``writer`` (a
    ``Writer``) and ``source_column``. ``table`` and ``writer`` are given together or not at all, or class creation
    raises ``TypeError``. The target is built by ``build_target`` when the class is defined, stored in
    ``__target__`` and read with ``target_of``. A subclass does not inherit a table: one that gives no keywords has
    ``__target__`` set to ``None``. A Part has no source column by default; a child gets the source with its parent's
    key. A subclass must be concrete: one with type parameters is refused with ``TypeError``. The ``container``
    keyword is only for a ``Record`` (see there); a Part that gives it is refused with ``TypeError``.

    Attributes:
        __target__ (ClassVar[Target | None]): The target built from the class keywords, or ``None`` when the class
            declares no table and is not a container ``Record``.

    """

    model_config = ConfigDict(extra="allow", frozen=True)  # unknown fields kept, so new ones can be reported
    __target__: ClassVar[Target | None] = None  # built when the class is defined; read it with target_of

    def __init_subclass__(
        cls,
        *,
        table: str | None = None,
        writer: Writer | None = None,
        source_column: str | None = None,
        container: bool = False,
        **kwargs: Any,
    ) -> None:
        """Keep the target keywords from ``BaseModel``.

        The keywords are used by ``__pydantic_init_subclass__``, which runs once the fields exist.

        Preconditions:
            None.

        Args:
            table: The target table; used by ``__pydantic_init_subclass__``.
            writer: The target writer; used by ``__pydantic_init_subclass__``.
            source_column: The source column; used by ``__pydantic_init_subclass__``.
            container: Whether the class is a container; used by ``__pydantic_init_subclass__``.
            **kwargs: Other class keyword arguments, passed on to ``BaseModel.__init_subclass__``.

        Returns:
            None.

        Raises:
            TypeError: From ``BaseModel.__init_subclass__``, for a class keyword it does not accept.

        """
        super().__init_subclass__(**kwargs)  # the keywords are used once the fields exist, below

    @classmethod
    def __pydantic_init_subclass__(
        cls,
        *,
        table: str | None = None,
        writer: Writer | None = None,
        source_column: str | None = None,  # a Record defaults to source_id; a child inherits it with its parent's key
        container: bool = False,  # only a Record may be one; Record takes it before calling this
        **kwargs: Any,
    ) -> None:
        """Build the new subclass's target from its class keywords and store it in ``__target__``.

        Pydantic calls this after it has built ``model_fields`` for the subclass. With neither ``table`` nor
        ``writer``, ``__target__`` is set to ``None``, so a table is not inherited. With both, ``__target__`` is
        ``build_target(cls, table, writer, source_column)``. A generic subclass (one with type parameters, e.g.
        ``class R[T](Record)`` or ``class R(Record, Generic[T])``) is refused, so a parametrised one (``R[int]``),
        which Pydantic would create with no class keywords and so without its origin's target, is never created.
        ``container`` reaches this as True only for a ``Part`` that is not a ``Record`` (``Record`` takes the keyword
        and builds its container itself), so it is refused. A ``TaskDay`` field is refused on a Part that is not a
        ``Record``, since only a record's own payload is filled.

        Preconditions:
            None.

        Args:
            table: The target table, ``"schema.table"`` in lower case, or ``None``.
            writer: How rows are written to ``table``, or ``None``.
            source_column: The column the source id is written to, or ``None`` for none.
            container: True to declare a container; refused here.
            **kwargs: Other class keyword arguments, passed on to ``BaseModel.__pydantic_init_subclass__``.

        Returns:
            None.

        Raises:
            TypeError: ``container`` is True (a Part cannot be a container); a field of a Part that is not a
                ``Record`` carries ``TaskDay``; the subclass is generic (has type parameters or is parametrised); only
                one of ``table`` and ``writer`` is given; or from
                ``build_target``, for a bad declaration (e.g. a field type that is not defined yet, a field without a
                ``To``, a writer that does not fit the columns). The checks run in that order.
            ValueError: From ``build_target``, for a bad declaration (e.g. a malformed table or column name, a
                writer key that names a column the table does not have).

        """
        super().__pydantic_init_subclass__(**kwargs)
        if container:
            raise TypeError(f"{cls.__name__}: only a Record can be a container; a Part loads with a table")
        marked = task_day_fields(cls)
        if marked and not issubclass(cls, Record):  # Record is defined below this class, before any marked one
            raise TypeError(f"{cls.__name__}.{marked[0]}: TaskDay fills a Record's own field, not a nested part's")
        generic = cls.__pydantic_generic_metadata__
        if generic["parameters"] or generic["origin"] is not None:  # R[int] would lose R's target
            raise TypeError(f"{cls.__name__}: a Record or Part must be concrete, not generic")
        if table is None and writer is None:
            cls.__target__ = None
        elif table is None or writer is None:
            raise TypeError(f"{cls.__name__}: table and writer are given together")
        else:
            cls.__target__ = build_target(cls, table, writer, source_column)


class Record(Part):
    """A ``Part`` whose subclasses must declare ``record_key`` and ``record_version``, checked at class creation.

    Every subclass, including an intermediate base, must have both class variables, set on itself or inherited, or
    class creation raises ``TypeError``; see ``__pydantic_init_subclass__``. ``Record`` itself is not checked and
    has neither attribute.

    A Record takes the same class keywords as ``Part``, but ``source_column`` defaults to ``"source_id"``. A Record with
    a table also has its writer key checked against ``record_key`` by ``_check_table_key``.

    A Record declared with ``container=True`` (``class ProductDay(Record, container=True)``) is a container: it has
    no ``table`` and no ``writer`` (giving either raises ``TypeError``) and fills only its ``Rows`` children's tables,
    of which it needs at least one. Its ``source_column`` keyword works as for any Record (``"source_id"`` by default,
    ``None`` for none). Its ``record_key`` fields, each with a ``To``, are the columns its children inherit, with the
    source column; no other field may have a ``To``. Its target is built by ``checks.targets.build_container`` once
    ``record_key`` is checked, which enforces these rules; ``_check_table_key`` is not run for it. A subclass that does
    not repeat ``container=True`` is not a container, as a subclass does not inherit a table.

    A key is read one way only, by ``key_of`` from the landed payload with ``_key_model``, never from a parsed record's
    attributes: Validate de-duplicates on it and Transform writes the target's key columns from it. Its validators,
    model validators and overrides do not reach the key. A ``field_validator`` or deprecated ``validator`` whose fields
    include a key field (by name, or ``"*"``) is refused, since its result would be the parsed field's value but never
    the key's; other decorator validators and overrides are accepted. A key that must be normalised (e.g. lower-cased)
    is expressed with the key field's declarative constraints (``StringConstraints``, ...), which ``_key_model`` carries
    and so apply identically to the key and to the parsed field. A key field's type and metadata must be allowed by
    ``_key_refusal``, so the key's values are hashable and storable. A subclass may not set ``strict=True`` in its
    ``model_config``: payloads are JSON-decoded values validated in Python mode, where strict validation refuses the
    string forms of ``date``, ``datetime``, ``UUID`` and ``Decimal``.

    A subclass may declare ``reshape`` (``reshape = flatten``, or with the ``ClassVar`` annotation), a pure function
    of one landed record that returns the record Validate parses and reads the key from, for a response whose shape
    the model cannot read as it is: lifting ``products[].skus[]`` into a list of its own, building child rows from
    named blocks. Validate applies it to each landed record that is a mapping, after the call's id is written in
    (``IdParam.inject``) and after the ``TaskDay`` fields are filled, and before the key is read and the record is
    parsed (``stages.validate``); a record whose ``reshape`` raises, or returns something other than a mapping, is
    held under ``RESHAPE_RULE``. The raw pages and a held record's stored payload stay as landed. It is inherited by a
    subclass; one that is not callable is refused at class creation.

    Attributes:
        record_key (ClassVar[tuple[str, ...]]): Names of the declared fields that identify a record, in key order. At
            least one; each must be a declared field that is required (no default), whose type is defined when the
            model is declared, that does not accept ``None`` as judged by ``_allows_none``, and whose type and
            metadata are allowed by ``_key_refusal``. Duplicate names are not rejected.
        record_version (ClassVar[int]): Version of the record's shape, 1 or more.
        reshape (ClassVar[Callable[[Mapping[str, Any]], Mapping[str, Any]]]): Optional. Turns a landed record into
            the record parsed; must not change its argument. Not set on ``Record``.

    """

    record_key: ClassVar[tuple[str, ...]]  # the fields that identify a record, e.g. ("id",)
    record_version: ClassVar[int]  # bumped when the model's meaning changes; stored with a schema failure
    reshape: ClassVar[Callable[[Mapping[str, Any]], Mapping[str, Any]]]  # optional; applied before parsing

    @classmethod
    def __pydantic_init_subclass__(
        cls, *, source_column: str | None = "source_id", container: bool = False, **kwargs: Any
    ) -> None:
        """Build the target, then check that the new subclass declares a usable ``record_key`` and ``record_version``.

        Pydantic calls this after it has built ``model_fields`` for the subclass. A container given a ``table`` or
        ``writer`` is refused before anything else. ``Part.__pydantic_init_subclass__`` then builds the target (``None``
        for a container, since it is given neither, and ``container`` is not passed on). Checks then run in order:
        ``model_config`` does not set ``strict=True``; every ``TaskDay`` field passes ``_check_task_days``; a
        ``reshape``, when set, is callable;
        ``record_key`` is non-empty, ``record_version`` is 1 or more,
        then for each key name in turn: it is a declared field, the field has no default, every item of its own metadata
        is allowed by ``_declarative``, no ``field_validator`` (any mode) or deprecated ``validator``, inherited ones
        included, names it or ``"*"``, its type can be resolved, it does not accept ``None`` as judged by
        ``_allows_none``, and its type, with the field's metadata, is allowed by ``_key_refusal`` (an allow-list of
        types and declarative metadata; ``float`` and every validator in metadata are refused, and so is ``strict``). A
        key field whose type Pydantic has no schema for without the model's config (a plain class under
        ``arbitrary_types_allowed=True``) is refused with ``_key_refusal``'s reason, which every such type has since
        every allowed form has a schema. Extra (undeclared) fields cannot be key fields. Last, a container's target is
        built by ``build_container`` and stored in ``__target__``, or else a Record with a target has its writer key
        checked by ``_check_table_key``. Other decorator validators (``model_validator``, ``root_validator``, a field
        validator on non-key fields), mixins and ``BaseModel`` overrides are not checked: the key is read by ``key_of``
        alone, which none of them reaches.

        A key field's own metadata is checked with ``_declarative`` before ``_allows_none`` runs, so a validator there
        is refused by name and never called with ``None``. The form given to ``_key_refusal`` comes from ``_annotated``.
        The list of field validators is rebuilt from ``__pydantic_decorators__`` once per key name, though it does not
        depend on the name.

        Preconditions:
            None.

        Args:
            source_column: The column the source id is written to, or ``None`` for none; passed on to
                ``Part.__pydantic_init_subclass__``, and to ``build_container`` for a container.
            container: True to declare a container (no table or writer, only ``Rows`` children); not passed on.
            **kwargs: Other class keyword arguments (``table``, ``writer``), passed on to
                ``Part.__pydantic_init_subclass__``.

        Returns:
            None.

        Raises:
            TypeError: ``model_config`` sets ``strict=True``, a ``TaskDay`` field breaks a rule of
                ``_check_task_days``, ``reshape`` is set to something that is not callable, ``record_key`` is missing
                or empty, ``record_version``
                is missing or less than 1, a key name is not a declared field, a key field has a default, a
                ``field_validator`` or deprecated ``validator`` names a key field or ``"*"`` (the message, the same for
                both, says key normalisation must be declarative), a key field's type names a class that is not defined
                yet (chained from the ``PydanticUserError`` raised by ``_allows_none``), a key field accepts ``None``, a
                key field's type or metadata is refused by ``_key_refusal`` (e.g. ``float``, ``list[str]``,
                ``str | int``, ``Literal[1, True]``, ``Literal["1", 1]``, a named tuple, a plain class under
                ``arbitrary_types_allowed=True``, a plain ``datetime``, a recursive ``type`` alias, an
                ``AfterValidator`` or ``BeforeValidator`` (refused before it can run), ``Field(strict=True)``, or a
                ``Decimal`` that accepts ``NaN``). Also raised by the ``<`` comparison when ``record_version`` is not
                comparable with ``int``; from ``_allows_none``, when Pydantic cannot build a key field's schema from its
                metadata; from ``_check_table_key``, when the writer key does not match ``record_key``; when a
                container is given a ``table`` or ``writer``; from ``build_container``, for a bad container
                declaration; and from ``Part.__pydantic_init_subclass__``, for a generic subclass or a bad target
                declaration.
            ValueError: From ``Part.__pydantic_init_subclass__``, for a bad target declaration; from
                ``build_container``, for a malformed source or ``To`` column or a ``Decimal`` precision or scale out
                of range.
            Exception: From ``_allows_none``, whatever a validator inside a key field's type (a ``type`` alias of an
                ``Annotated`` form, say), not in its own metadata, raises on ``None`` other than ``ValueError`` or
                ``AssertionError``.

        """
        if container and (kwargs.get("table") is not None or kwargs.get("writer") is not None):
            raise TypeError(f"{cls.__name__}: a container has no table or writer; only its Rows children load")
        super().__pydantic_init_subclass__(source_column=source_column, **kwargs)
        if cls.model_config.get("strict"):  # Python-mode strict refuses the JSON string forms of dates, UUIDs, ...
            raise TypeError(f"{cls.__name__}: a Record cannot set strict=True, since its payloads are decoded JSON")
        _check_task_days(cls)
        if hasattr(cls, "reshape") and not callable(cls.reshape):
            raise TypeError(f"{cls.__name__}.reshape must be a function of one record, not {cls.reshape!r}")
        key = getattr(cls, "record_key", ())
        if not key:
            raise TypeError(f"{cls.__name__} must set record_key")
        if getattr(cls, "record_version", 0) < 1:
            raise TypeError(f"{cls.__name__} must set record_version to 1 or more")
        for name in key:
            field = cls.model_fields.get(name)
            if field is None:
                raise TypeError(f"{cls.__name__}.record_key names {name!r}, which is not a field")
            if not field.is_required():
                raise TypeError(f"{cls.__name__}.{name} is in record_key, so it cannot have a default")
            refused = next((item for item in field.metadata if not _declarative(item)), None)
            if refused is not None:  # before _allows_none, which would call a validator with None
                raise TypeError(f"{cls.__name__}.{name} is in record_key, so it cannot have the metadata {refused!r}")
            decorators = cls.__pydantic_decorators__
            validators = [(v.cls_var_name, v.info.fields) for v in decorators.field_validators.values()]
            validators += [(v.cls_var_name, v.info.fields) for v in decorators.validators.values()]  # deprecated
            validator = next((var for var, fields in validators if name in fields or "*" in fields), None)
            if validator is not None:  # key_of never runs it, so the key would differ from the parsed field
                raise TypeError(
                    f"{cls.__name__}.{name} is in record_key, so it cannot have the field_validator "
                    f"{validator!r}; normalise a key declaratively, e.g. StringConstraints(to_lower=True)"
                )
            form = _annotated(field)
            try:
                allows_none = _allows_none(field)
            except PydanticSchemaGenerationError as err:  # a class allowed only by arbitrary_types_allowed
                reason = _key_refusal(form, (), cls.model_config)  # not None: every allowed form has a schema
                raise TypeError(f"{cls.__name__}.{name} is in record_key, so {reason}") from err
            except PydanticUserError as err:  # the type names a class that is not defined yet
                raise TypeError(f"{cls.__name__}.{name} is in record_key, so its type must be defined first") from err
            if allows_none:
                raise TypeError(f"{cls.__name__}.{name} is in record_key, so it cannot allow None")
            reason = _key_refusal(form, (), cls.model_config)
            if reason is not None:
                raise TypeError(f"{cls.__name__}.{name} is in record_key, so {reason}")
        if container:  # its key fields are what its children inherit, so it is built once they are checked
            cls.__target__ = build_container(cls, key, source_column)
        elif cls.__target__ is not None:
            _check_table_key(cls.__name__, cls.__target__, key)


def _check_task_days(model: type[BaseModel]) -> None:
    """Check that every ``TaskDay`` field of a model is a Record's plain ``int`` field with no alias.

    Each field marked ``TaskDay`` must carry one ``TaskDay``, have the type ``int`` (Pydantic moves an ``Int32``'s
    bounds and ``Width`` into the field's metadata, so ``Int32`` is ``int`` here; an optional ``int`` or a ``type``
    alias is refused), and have neither an ``alias`` nor a ``validation_alias``, since Validate writes the day under
    the field's name.

    Preconditions:
        ``model`` is a ``Record`` subclass; ``Part.__pydantic_init_subclass__`` refuses a marked field of any other
        model.

    Args:
        model: The record model whose fields are checked.

    Returns:
        None.

    Raises:
        TypeError: A marked field carries more than one ``TaskDay``, is not of type ``int``, or has an alias.

    """
    for name in task_day_fields(model):
        field, where = model.model_fields[name], f"{model.__name__}.{name}"
        if sum(isinstance(item, TaskDay) for item in field.metadata) > 1:
            raise TypeError(f"{where}: a field takes one TaskDay mark")
        if field.annotation is not int:
            raise TypeError(
                f"{where}: TaskDay fills a YYYYMMDD int, so the field must be an int, not {field.annotation!r}"
            )
        if field.alias is not None or field.validation_alias is not None:
            raise TypeError(f"{where}: TaskDay writes the day under the field's name, so the field takes no alias")


def _check_table_key(name: str, target: Target, key: tuple[str, ...]) -> None:
    """Check that a Record's writer key is its source column plus its ``record_key`` columns.

    Only an ``Upsert``, ``Snapshot`` or ``Update`` writer is checked; other writers return at once. Every
    ``record_key`` field needs a ``To``, and the writer's key must be exactly the source column (when the target has
    one) plus those fields' columns, in any order. Validate keeps one record per ``record_key``, so this leaves no
    two rows with one key in a batch.

    Preconditions:
        ``target`` was built by ``build_target`` for the Record named ``name``, and ``key`` is its checked
        ``record_key``.

    Args:
        name: The Record's class name, used in error messages.
        target: The Record's target.
        key: The Record's ``record_key``.

    Returns:
        None.

    Raises:
        TypeError: A ``record_key`` field has no ``To``, or the writer's key is not exactly the expected columns.

    """
    # Validate keeps one record per record_key, so a table key built from it is unique in a batch
    if not isinstance(target.writer, Upsert | Snapshot | Update):
        return
    columns = {column.path: column.name for column in target.columns}
    for field in key:
        if (field,) not in columns:
            raise TypeError(f"{name}.{field} is in record_key, so it needs a To")
    source = {target.source_column} if target.source_column is not None else set()
    expected = source | {columns[(field,)] for field in key}
    if set(target.writer.key) != expected:
        raise TypeError(f"{name}: the writer's key must be the columns {', '.join(sorted(expected))}")


def _annotated(field: FieldInfo) -> Any:
    """Return the field's type annotated with its own metadata, as a key field's form is checked.

    Preconditions:
        None.

    Args:
        field: The ``FieldInfo`` Pydantic built for the field.

    Returns:
        ``Annotated[annotation, *metadata]`` when ``field.metadata`` is non-empty, else ``field.annotation`` itself
        (``Annotated`` needs at least one metadata item).

    Raises:
        Nothing.

    """
    return Annotated[(field.annotation, *field.metadata)] if field.metadata else field.annotation


def _allows_none(field: FieldInfo) -> bool:
    """Return whether a field accepts ``None``.

    Validates ``None`` against ``_annotated(field)``, ``field.annotation`` annotated with ``field.metadata``, so the
    field's constraints and the validators in its ``Annotated`` metadata apply. The field's other settings, its aliases
    among them, are not passed, so Pydantic does not warn that they have no effect outside a model. Any form that
    accepts ``None`` is caught, including ``object``, ``Literal[None]``, a type alias and an ``Annotated`` member of a
    union. ``Record`` runs this before ``_key_refusal``, which refuses every validator in a key field's metadata and
    every type that accepts ``None``, so this gives those types their own message and finds a type that is not defined
    yet.

    Preconditions:
        None.

    Args:
        field: The ``FieldInfo`` Pydantic built for the field.

    Returns:
        ``True`` if validation succeeds and the result is still ``None``; ``False`` if it raises ``ValidationError`` or
        a validator turns ``None`` into another value.

    Raises:
        PydanticUserError: The field's type names a class that is not defined yet, or (as
            ``PydanticSchemaGenerationError``) a class Pydantic has no schema for, since the model's config, its
            ``arbitrary_types_allowed`` among them, is not passed; ``Record`` turns either into ``TypeError``, the
            second with ``_key_refusal``'s reason.
        TypeError: Pydantic cannot build a schema from the field's metadata (e.g. ``Field(discriminator=...)`` on a
            ``str``); ``Record`` lets it propagate, though Pydantic's own model build, which runs first, already
            refuses that example.
        Exception: Whatever a validator in the field's metadata raises other than ``ValueError`` or
            ``AssertionError``, which Pydantic turns into ``ValidationError``.

    """
    try:
        return TypeAdapter[Any](_annotated(field)).validate_python(None) is None
    except ValidationError:
        return False


def _key_refusal(
    form: object, metadata: tuple[object, ...], config: ConfigDict, aliases: tuple[object, ...] = ()
) -> str | None:
    """Return why a type form is refused for a key field, or ``None`` when it is allowed.

    Allowed forms: ``str``, ``int``, ``bool``, ``date``, ``AwareDatetime``, ``Decimal``, ``UUID``, a ``StrEnum`` or
    ``IntEnum`` subclass, a ``Literal`` whose values are all ``str``, all ``int`` or all ``bool`` (a mix is refused:
    ``1 == True``, so two records would share a key, and a key compared across types is ambiguous), and a
    fixed-length ``tuple[...]`` (at least one item, no ``...``) whose items are each allowed. ``Annotated[X, ...]`` is
    judged as ``X`` when every metadata item is declarative, as judged by ``_declarative``; a ``type X = ...`` alias,
    bare or subscripted, as its value from ``alias_value`` (an alias whose arguments cannot be substituted is
    refused, and so is an alias met again inside its own value, such as ``type T = tuple[int, T]``); and a
    ``NewType`` as its supertype. A ``Decimal`` is refused when it accepts ``Decimal("NaN")`` with the
    metadata on its way and the model's ``allow_inf_nan`` setting. A plain ``datetime`` is refused with a reason naming
    ``AwareDatetime``, since it accepts naive values. Every other form is refused: ``float``, a union
    (``Optional`` among them), ``Any``, ``object``, a bare ``tuple``, a ``list``, ``dict``, ``set``, ``frozenset``, a
    named tuple, a dataclass, a model and any other class or generic alias. Any ``StrEnum`` or ``IntEnum`` subclass is
    allowed whatever it defines, a ``__get_pydantic_core_schema__`` that replaces Pydantic's enum validation included;
    ``key_of`` reads the key from the payload with ``_key_model`` alone, so such a class cannot give a failed copy and
    a parsed copy of one payload different keys.

    Preconditions:
        ``form`` is resolved: ``_allows_none`` has built it without a ``PydanticUserError``, or with a
        ``PydanticSchemaGenerationError`` (every class it names is defined).

    Args:
        form: A type form, as Pydantic stores it in ``FieldInfo.annotation``.
        metadata: The metadata that applies to ``form`` itself: the field's metadata at the top, then that of each
            ``Annotated`` around it; an item of a tuple starts with none.
        config: The model's config, read for ``allow_inf_nan``.
        aliases: The ``type`` alias forms (bare or subscripted) being judged around ``form``, outermost first; a
            caller passes none.

    Returns:
        ``None`` when the form is allowed, otherwise the reason it is refused, naming the refused form, metadata item
        or ``NaN``, to follow ``"so "`` in an error message.

    Raises:
        Nothing: A ``TypeError`` from ``alias_value`` is caught and the form refused.

    """
    if isinstance(form, TypeAliasType) or isinstance(get_origin(form), TypeAliasType):  # type X = ..., or X[int]
        if form in aliases:  # met inside its own value, so it would be followed without end
            return f"its type cannot be the recursive alias {form!r}"
        try:
            value = alias_value(form)
        except TypeError:  # the arguments cannot be substituted, so the value is unknown
            return f"its type cannot be {form!r}"
        return _key_refusal(value, metadata, config, (*aliases, form))
    if isinstance(form, NewType):
        return _key_refusal(form.__supertype__, metadata, config, aliases)
    origin, args = get_origin(form), get_args(form)
    if origin is Annotated:
        extra = form.__metadata__  # type: ignore[attr-defined]
        refused = next((item for item in extra if not _declarative(item)), None)
        if refused is not None:
            return f"it cannot have the metadata {refused!r}"
        return _key_refusal(args[0], (*metadata, *extra), config, aliases)
    if origin is Literal:
        kinds = {type(value) for value in args}
        if len(kinds) == 1 and kinds <= {str, int, bool}:  # one type only: 1 == True, and "1" is not 1
            return None
        return f"its type cannot be {form!r}"
    if origin is tuple and args and Ellipsis not in args:  # a bare tuple or typing.Tuple has no args
        return next((reason for arg in args if (reason := _key_refusal(arg, (), config, aliases)) is not None), None)
    if form is Decimal:
        nan_config = ConfigDict(allow_inf_nan=config["allow_inf_nan"]) if "allow_inf_nan" in config else None
        adapter = TypeAdapter[Any](Annotated[(Decimal, *metadata)] if metadata else Decimal, config=nan_config)
        try:
            adapter.validate_python(Decimal("NaN"))
        except ValidationError:
            return None
        return "it cannot accept NaN"
    if form is datetime:
        return "its type cannot be a plain datetime, which accepts naive values: use pydantic's AwareDatetime"
    if form in _KEY_CLASSES or (isinstance(form, type) and issubclass(form, StrEnum | IntEnum)):
        return None
    return f"its type cannot be {form!r}"


def _declarative(item: object) -> bool:
    """Return whether a metadata item is a declarative constraint or mark, which a key field may carry.

    A key field's metadata is carried into ``_key_model``, so it is what reads the key; a validator there could give a
    value outside the types ``_key_refusal`` allows (unhashable or unstorable), and ``Strict`` refuses the JSON
    string forms of dates, UUIDs and decimals, so the key could not be read.

    Allowed: an ``annotated_types`` constraint (``BaseMetadata``: ``MinLen``, ``Ge`` and Pydantic's own, such as the
    general metadata ``Field(...)`` leaves), Pydantic's ``AllowInfNan``, ``Strict(False)``, a group of them
    (``GroupedMetadata``: ``StringConstraints``, ``Interval``) whose items are each allowed, a ``FieldInfo``
    (``Field(...)`` inside an ``Annotated`` that is not the field's own) whose metadata items are each allowed, a ``To``
    mark, a ``Width`` mark (so ``Int32`` and ``Int16`` keys are allowed; Pydantic ignores the mark when it validates), a
    ``TaskDay`` mark, which Validate fills before the key is read, and the ``_DateKeyInput`` mark of ``DateKey``, whose
    fixed conversion gives only a ``YYYYMMDD`` int. Anything else is refused: ``Strict(True)`` (``Field(strict=True)``,
    ``StrictStr``, ...), ``annotated_types.Predicate``, which runs code, and every validator (``AfterValidator``,
    ``BeforeValidator``, ``WrapValidator``, ``PlainValidator``) among them. The test is ``isinstance``, so an instance
    of any subclass of ``BaseMetadata``, ``AllowInfNan``, ``GroupedMetadata``, ``To``, ``Width``, ``TaskDay`` or
    ``_DateKeyInput`` is allowed, including one whose class defines ``__get_pydantic_core_schema__`` and so runs code of
    its own; ``key_of`` reads the key from the payload with ``_key_model`` alone, so such an item cannot give a failed
    copy and a parsed copy of one payload different keys.

    Preconditions:
        None.

    Args:
        item: One metadata item of an ``Annotated`` form or a field.

    Returns:
        ``True`` when the item is allowed, otherwise ``False``.

    Raises:
        Nothing.

    """
    if isinstance(item, FieldInfo):
        return all(_declarative(inner) for inner in item.metadata)
    if isinstance(item, GroupedMetadata):
        return all(_declarative(inner) for inner in item)
    if isinstance(item, Strict):
        return not item.strict  # strict Python mode refuses the JSON string forms the key arrives in
    return isinstance(item, BaseMetadata | AllowInfNan | To | Width | TaskDay | _DateKeyInput) and not isinstance(
        item, Predicate
    )  # Predicate runs code


@dataclass(frozen=True, slots=True)
class Keyed[R: Record]:
    """A parsed record together with its key from ``key_of``.

    Frozen and slotted. No field is checked; Validate builds every instance from one landed payload.

    Attributes:
        record: The parsed record; Transform writes every column but the key columns from it.
        key: ``key_of`` of the record's model and the same payload, in ``record_key`` order; Validate de-duplicates
            on it and Transform writes the target's key columns from it.

    """

    record: R
    key: tuple[object, ...]  # read from the payload, never from record's attributes


def key_of(model: type[Record], payload: Any) -> tuple[object, ...]:
    """Return a record's key values in ``record_key`` order, read from its landed payload.

    This is the only way a key is read: ``payload`` is validated against ``_key_model(model)`` and the key fields'
    values are returned as that model gives them, never read from a parsed record's attributes. So a copy that parsed
    and a copy that failed to parse get the same key from the same payload.

    Preconditions:
        ``model`` is a subclass of ``Record``, so ``__pydantic_init_subclass__`` has checked ``record_key``. Every
        caller passes a job's record model.

    Args:
        model: The record class whose key is read.
        payload: The record as landed; any JSON value.

    Returns:
        One value per name in ``model.record_key``, normalised by each key field's type and constraints (e.g. a
        ``UUID`` as a ``UUID``, a lower-cased ``str`` for a ``to_lower`` constraint).

    Raises:
        ValidationError: The key fields do not validate (e.g. ``payload`` is not an object, or a key field is absent,
            ``null`` or of the wrong type).
        AttributeError: ``model`` is ``Record`` itself, which has no ``record_key``.
        Exception: Whatever a metadata class or enum with its own ``__get_pydantic_core_schema__`` raises while a key
            field is validated, other than ``ValueError`` or ``AssertionError``.

    """
    found = _key_model(model).model_validate(payload)
    return tuple(getattr(found, name) for name in model.record_key)


@cache
def _key_model(model: type[Record]) -> type[BaseModel]:
    """Return a model of only a Record's ``record_key`` fields, built once per record model.

    The model is named ``<model>Key`` and has ``model``'s ``model_config`` with ``extra="ignore"``, so the payload's
    other fields are ignored even when ``model`` forbids extra fields. For each key field it has the field's annotation
    and ``FieldInfo`` (aliases, ``AliasChoices`` and ``AliasPath`` included, and its constraint metadata, such as
    ``StringConstraints(to_lower=True)``, which so normalise the key and the parsed field alike). No decorator
    validator, model validator or override of ``model`` is copied, so none of them changes the key.

    Preconditions:
        ``model`` is a subclass of ``Record``, so every ``record_key`` name is a declared field whose metadata
        ``_key_refusal`` allows.

    Args:
        model: The record class.

    Returns:
        The key-only model class.

    Raises:
        AttributeError: ``model`` is ``Record`` itself, which has no ``record_key``.

    """
    key = model.record_key
    fields: dict[str, Any] = {name: (model.model_fields[name].annotation, model.model_fields[name]) for name in key}
    config = ConfigDict(**model.model_config) | ConfigDict(extra="ignore")
    return create_model(f"{model.__name__}Key", __config__=config, **fields)
