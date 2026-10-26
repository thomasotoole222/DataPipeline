"""Base models for parsed connector records, the check that each record type declares a usable key, and key lookup.

Public API:
    Part: Frozen pydantic model that keeps unknown fields; the base for nested parts of a record.
    Record: A ``Part`` whose subclasses must declare ``record_key`` and ``record_version``, checked at class creation;
        each key field must be required, have a type defined when the model is declared, and not accept ``None``.
    key_of: Return a record's key values in ``record_key`` order.

A model may declare its target table, writer and source column as class keywords
(``class Order(Record, table="sales.order", writer=Upsert(...))``); the target is built by ``build_target``
(checks.targets) when the class is defined.

Private units:
    _allows_none: Return whether a field accepts ``None``.
    _check_table_key: Check that a Record's writer key is its source column plus its ``record_key`` columns.
"""

from typing import Annotated, Any, ClassVar

from pydantic import BaseModel, ConfigDict, PydanticUserError, TypeAdapter, ValidationError
from pydantic.fields import FieldInfo

from pipeline.checks.targets import Snapshot, Target, Update, Upsert, Writer, build_target


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
    key.

    Attributes:
        __target__ (ClassVar[Target | None]): The target built from the class keywords, or ``None`` when the class
            declares no table.

    """

    model_config = ConfigDict(extra="allow", frozen=True)  # unknown fields kept, so new ones can be reported
    __target__: ClassVar[Target | None] = None  # built when the class is defined; read it with target_of

    def __init_subclass__(
        cls,
        *,
        table: str | None = None,
        writer: Writer | None = None,
        source_column: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Accept the target keywords so they are not passed on to ``BaseModel.__init_subclass__``.

        The keywords are used by ``__pydantic_init_subclass__``, which runs once the fields exist.

        Preconditions:
            None.

        Args:
            table: The target table; used by ``__pydantic_init_subclass__``.
            writer: The target writer; used by ``__pydantic_init_subclass__``.
            source_column: The source column; used by ``__pydantic_init_subclass__``.
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
        **kwargs: Any,
    ) -> None:
        """Build the new subclass's target from its class keywords and store it in ``__target__``.

        Pydantic calls this after it has built ``model_fields`` for the subclass. With neither ``table`` nor
        ``writer``, ``__target__`` is set to ``None``, so a table is not inherited. With both, ``__target__`` is
        ``build_target(cls, table, writer, source_column)``.

        Preconditions:
            None.

        Args:
            table: The target table, ``"schema.table"`` in lower case, or ``None``.
            writer: How rows are written to ``table``, or ``None``.
            source_column: The column the source id is written to, or ``None`` for none.
            **kwargs: Other class keyword arguments, passed on to ``BaseModel.__pydantic_init_subclass__``.

        Returns:
            None.

        Raises:
            TypeError: Only one of ``table`` and ``writer`` is given; or from ``build_target``, for a bad
                declaration (e.g. a field type that is not defined yet, a field without a ``To``, a writer that does
                not fit the columns).
            ValueError: From ``build_target``, for a bad declaration (e.g. a malformed table or column name, a
                writer key that names a column the table does not have).

        """
        super().__pydantic_init_subclass__(**kwargs)
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
    a target also has its writer key checked against ``record_key`` by ``_check_table_key``.

    Attributes:
        record_key (ClassVar[tuple[str, ...]]): Names of the declared fields that identify a record, in key order. At
            least one; each must be a declared field that is required (no default), whose type is defined when the
            model is declared, and that does not accept ``None`` as judged by ``_allows_none``. Duplicate names are
            not rejected.
        record_version (ClassVar[int]): Version of the record's shape, 1 or more.

    """

    record_key: ClassVar[tuple[str, ...]]  # the fields that identify a record, e.g. ("id",)
    record_version: ClassVar[int]  # bumped when the model's meaning changes; stored with a schema failure

    def __init_subclass__(cls, *, source_column: str | None = "source_id", **kwargs: Any) -> None:
        """Pass the class keywords on to ``Part.__init_subclass__``; ``source_column`` defaults to ``"source_id"``.

        Preconditions:
            None.

        Args:
            source_column: The source column; used by ``__pydantic_init_subclass__``.
            **kwargs: Other class keyword arguments, passed on to ``Part.__init_subclass__``.

        Returns:
            None.

        Raises:
            TypeError: From ``BaseModel.__init_subclass__``, for a class keyword it does not accept.

        """
        super().__init_subclass__(source_column=source_column, **kwargs)

    @classmethod
    def __pydantic_init_subclass__(cls, *, source_column: str | None = "source_id", **kwargs: Any) -> None:
        """Build the target, then check that the new subclass declares a usable ``record_key`` and ``record_version``.

        Pydantic calls this after it has built ``model_fields`` for the subclass. ``Part.__pydantic_init_subclass__``
        builds the target first. Checks then run in order: ``record_key`` is non-empty, ``record_version`` is 1 or
        more, then for each key name in turn: it is a declared field, the field has no default, its type can be
        resolved, and it does not accept ``None`` as judged by ``_allows_none``. Extra (undeclared) fields cannot be
        key fields. Last, a Record with a target has its writer key checked by ``_check_table_key``.

        Preconditions:
            None.

        Args:
            source_column: The column the source id is written to, or ``None`` for none; passed on to
                ``Part.__pydantic_init_subclass__``.
            **kwargs: Other class keyword arguments (``table``, ``writer``), passed on to
                ``Part.__pydantic_init_subclass__``.

        Returns:
            None.

        Raises:
            TypeError: ``record_key`` is missing or empty, ``record_version`` is missing or less than 1, a key name is
                not a declared field, a key field has a default, a key field's type names a class that is not
                defined yet (chained from the ``PydanticUserError`` raised by ``_allows_none``), or a key field
                accepts ``None``. Also raised by the ``<`` comparison when ``record_version`` is not comparable with
                ``int``; from ``_check_table_key``, when the writer key does not match ``record_key``; and from
                ``Part.__pydantic_init_subclass__``, for a bad target declaration.
            ValueError: From ``Part.__pydantic_init_subclass__``, for a bad target declaration.

        """
        super().__pydantic_init_subclass__(source_column=source_column, **kwargs)
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
            try:
                allows_none = _allows_none(field)
            except PydanticUserError as err:  # the type names a class that is not defined yet
                raise TypeError(f"{cls.__name__}.{name} is in record_key, so its type must be defined first") from err
            if allows_none:
                raise TypeError(f"{cls.__name__}.{name} is in record_key, so it cannot allow None")
        if cls.__target__ is not None:
            _check_table_key(cls.__name__, cls.__target__, key)


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


def _allows_none(field: FieldInfo) -> bool:
    """Return whether a field accepts ``None``.

    Validates ``None`` against ``Annotated[field.annotation, field]``, so the field's constraints and the validators
    in its ``Annotated`` metadata apply. Any form that accepts ``None`` is caught, including ``object``,
    ``Literal[None]``, a type alias and an ``Annotated`` member of a union.

    Preconditions:
        None.

    Args:
        field: The ``FieldInfo`` Pydantic built for the field.

    Returns:
        ``True`` if validation succeeds and the result is still ``None``; ``False`` if it raises ``ValidationError`` or
        a validator turns ``None`` into another value.

    Raises:
        PydanticUserError: The field's type names a class that is not defined yet; ``Record`` turns it into
            ``TypeError``.
        Exception: Whatever a validator in the field's metadata raises other than ``ValueError`` or
            ``AssertionError``, which Pydantic turns into ``ValidationError``.

    """
    form: Any = Annotated[field.annotation, field]  # the field's type with its constraints and validators
    try:
        return TypeAdapter[Any](form).validate_python(None) is None
    except ValidationError:
        return False


def key_of(record: Record) -> tuple[object, ...]:
    """Return a record's key values in ``record_key`` order.

    Values are read with ``getattr`` and returned as they are, not copied or converted.

    Preconditions:
        ``type(record)`` is a subclass of ``Record``, so ``__pydantic_init_subclass__`` has checked ``record_key``.
        Every caller passes a job's record model.

    Args:
        record: The record to read the key from.

    Returns:
        One value per name in ``type(record).record_key``.

    Raises:
        AttributeError: ``record`` is an instance of ``Record`` itself, which has no ``record_key``, or an instance
            built without validation (e.g. ``model_construct``) that lacks a key field.

    """
    return tuple(getattr(record, name) for name in type(record).record_key)
