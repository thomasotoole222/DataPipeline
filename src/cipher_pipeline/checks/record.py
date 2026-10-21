"""Base models for parsed source records, the check that each record type declares a usable key, and key lookup.

Public API:
    Part: Frozen pydantic model that keeps unknown fields; the base for nested parts of a record.
    Record: A ``Part`` whose subclasses must declare ``record_key`` and ``record_version``, checked at class creation;
        each key field must be required, have a type defined when the model is declared, and not accept ``None``.
    key_of: Return a record's key values in ``record_key`` order.
"""

from typing import Annotated, Any, ClassVar

from pydantic import BaseModel, ConfigDict, PydanticUserError, TypeAdapter, ValidationError
from pydantic.fields import FieldInfo


class Part(BaseModel):
    """Frozen pydantic model that keeps unknown fields; the base for nested parts of a record.

    ``extra="allow"`` keeps input keys that are not declared fields as extra attributes instead of rejecting or
    dropping them. ``frozen=True`` makes assignment to fields raise and gives instances a ``__hash__`` over the
    declared field values; extra values are not hashed.

    """

    model_config = ConfigDict(extra="allow", frozen=True)  # unknown fields kept, so new ones can be reported


class Record(Part):
    """A ``Part`` whose subclasses must declare ``record_key`` and ``record_version``, checked at class creation.

    Every subclass, including an intermediate base, must have both class variables, set on itself or inherited, or
    class creation raises ``TypeError``; see ``__pydantic_init_subclass__``. ``Record`` itself is not checked and
    has neither attribute.

    Attributes:
        record_key (ClassVar[tuple[str, ...]]): Names of the declared fields that identify a record, in key order. At
            least one; each must be a declared field that is required (no default), whose type is defined when the
            model is declared, and that does not accept ``None`` as judged by ``_allows_none``. Duplicate names are
            not rejected.
        record_version (ClassVar[int]): Version of the record's shape, 1 or more.

    """

    record_key: ClassVar[tuple[str, ...]]  # the fields that identify a record, e.g. ("id",)
    record_version: ClassVar[int]  # bumped when the model's meaning changes; stored with a schema failure

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Check that the new subclass declares a usable ``record_key`` and ``record_version``.

        Pydantic calls this after it has built ``model_fields`` for the subclass. Checks run in order: ``record_key``
        is non-empty, ``record_version`` is 1 or more, then for each key name in turn: it is a declared field, the
        field has no default, its type can be resolved, and it does not accept ``None`` as judged by
        ``_allows_none``. Extra (undeclared) fields cannot be key fields.

        Preconditions:
            None.

        Args:
            **kwargs: Class keyword arguments, passed on to ``BaseModel.__pydantic_init_subclass__``.

        Returns:
            None.

        Raises:
            TypeError: ``record_key`` is missing or empty, ``record_version`` is missing or less than 1, a key name is
                not a declared field, a key field has a default, a key field's type names a class that is not
                defined yet (chained from the ``PydanticUserError`` raised by ``_allows_none``), or a key field
                accepts ``None``. Also raised by the ``<`` comparison when ``record_version`` is not comparable with
                ``int``.

        """
        super().__pydantic_init_subclass__(**kwargs)
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
