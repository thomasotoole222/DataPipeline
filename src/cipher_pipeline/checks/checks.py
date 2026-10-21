"""Declarations of the checks run on parsed records, and the policy that decides when a batch fails.

Nothing here runs a check: the types only hold and validate declarations for the code that runs them.

Public API:
    SCHEMA_RULE: The name reserved for the schema check, ``"schema"``; no declared check may use it.
    Severity: What a failed check means: ``error``, ``warn`` or ``wait``.
    CheckContext: The job, shop, partition and time frame a check runs in.
    Rule: A named predicate over one record, validated on construction.
    Lookup: A named check that a record's values exist in a column of another table, validated on construction.
    BatchPolicy: The share and minimum count of held rows at which a batch fails.
    Validation: A record model with its uniquely named checks and the batch policy that applies to them.
"""

import re
from collections.abc import Callable, Iterable
from dataclasses import KW_ONLY, dataclass, field
from enum import StrEnum
from fractions import Fraction
from typing import Final
from uuid import UUID

from cipher_pipeline.checks.record import Record
from cipher_pipeline.timeutil import TimeFrame

#: The rule name a failed parse is held under; no check may use it.
SCHEMA_RULE: Final = "schema"

#: A ``schema.table`` name: two lower-case identifiers of ASCII letters, digits and ``_``, not starting with a digit,
#: joined by one dot. Used with ``fullmatch``.
_TABLE: Final = re.compile(r"[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*")
#: A bare column name: one lower-case identifier of ASCII letters, digits and ``_``, not starting with a digit. Used
#: with ``fullmatch``.
_COLUMN: Final = re.compile(r"[a-z_][a-z0-9_]*")


class Severity(StrEnum):
    """What a failed check means. Each member's value is its lower-case name.

    On this page only ``WAIT`` carries a rule: a check has an ``until`` job exactly when its severity is ``WAIT``
    (see ``_check_declaration``). What ``ERROR`` and ``WARN`` do is decided by the code that runs the checks.

    Attributes:
        ERROR: ``"error"``. The default severity of a ``Rule``.
        WARN: ``"warn"``.
        WAIT: ``"wait"``. The default severity of a ``Lookup``; requires ``until`` to be set.

    """

    ERROR = "error"  # held; a person resolves it
    WARN = "warn"  # loads; counted and logged
    WAIT = "wait"  # held until the job named in `until` next succeeds for the shop


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckContext:
    """The job, shop, partition and time frame a check runs in. Frozen and keyword-only; no field is checked.

    Attributes:
        job: The name of the job running the checks.
        shop_id: The shop whose records are checked.
        partition_key: The key of the partition being checked. An empty string is not rejected.
        frame: The time frame the records were fetched for, or ``None`` when there is none.

    """

    job: str
    shop_id: UUID
    partition_key: str
    frame: TimeFrame | None  # the task's time window; None for a job without windows


@dataclass(frozen=True, slots=True)
class Rule[R: Record]:
    """A named predicate over one record, with its version and severity, validated on construction.

    ``name`` and ``check`` may be positional; the other fields are keyword-only. Frozen.

    Invariants, all enforced by ``__post_init__`` through ``_check_declaration``:
        ``name`` is not empty and is not ``SCHEMA_RULE``.
        ``version`` is 1 or more.
        ``until`` is not ``None`` exactly when ``severity`` is ``Severity.WAIT``, and is then not ``""``.

    Attributes:
        name: The check's name, unique within a ``Validation``.
        check: The predicate, called with a record and its ``CheckContext``. Nothing on this page calls it or fixes
            what ``True`` means.
        version: The version of the check, 1 or more. Defaults to 1.
        severity: What a failure means. Defaults to ``Severity.ERROR``.
        overridable: Stored only; not used on this page. Defaults to ``False``.
        until: The job a ``WAIT`` check waits for; ``None`` for any other severity. Defaults to ``None``.

    """

    name: str
    check: Callable[[R, CheckContext], bool]  # True when the record passes
    _: KW_ONLY
    version: int = 1  # bumped when the check's meaning changes; stored with each failure
    severity: Severity = Severity.ERROR
    overridable: bool = False  # whether a person may release a record held by it
    until: str | None = None  # the job a waiting record waits for; set exactly when severity is WAIT

    def __post_init__(self) -> None:
        """Reject a declaration that breaks the class invariants.

        Preconditions:
            None.

        Returns:
            None.

        Raises:
            ValueError: From ``_check_declaration``, if ``name``, ``version``, ``severity`` or ``until`` breaks an
                invariant.

        """
        _check_declaration(self.name, self.version, self.severity, self.until)


@dataclass(frozen=True, slots=True)
class Lookup[R: Record]:
    """A named check that a record's values exist in a column of another table, validated on construction.

    ``name`` and ``values`` may be positional; the other fields are keyword-only. Frozen.

    ``severity`` defaults to ``Severity.WAIT``, so a lookup must give ``until`` or set another severity; one that does
    neither raises ``ValueError`` when it is declared. This is deliberate: the framework cannot guess which job a
    lookup waits for, so the author must name it.

    Invariants, all enforced by ``__post_init__``:
        ``name`` is not empty and is not ``SCHEMA_RULE``.
        ``version`` is 1 or more.
        ``until`` is not ``None`` exactly when ``severity`` is ``Severity.WAIT``, and is then not ``""``.
        ``table`` matches ``_TABLE``: a lower-case ``schema.table``.
        ``column`` matches ``_COLUMN``: a lower-case bare column name.

    Attributes:
        name: The check's name, unique within a ``Validation``.
        values: Returns the values of a record to look up. Nothing on this page calls it.
        table: The ``schema.table`` to look the values up in.
        column: The column of ``table`` that holds the values.
        version: The version of the check, 1 or more. Defaults to 1.
        severity: What a failure means. Defaults to ``Severity.WAIT``.
        overridable: Stored only; not used on this page. Defaults to ``False``.
        until: The job a ``WAIT`` check waits for; ``None`` for any other severity. Defaults to ``None``.

    """

    name: str
    values: Callable[[R], Iterable[str]]  # the values the record refers to
    _: KW_ONLY
    table: str  # "schema.table"
    column: str  # the column the values must appear in, compared as text
    version: int = 1  # bumped when the check's meaning changes; stored with each failure
    severity: Severity = Severity.WAIT  # a missing reference usually arrives with a later job
    overridable: bool = False  # whether a person may release a record held by it
    until: str | None = None  # the job a waiting record waits for; set exactly when severity is WAIT

    def __post_init__(self) -> None:
        """Reject a declaration that breaks the class invariants.

        The shared checks run first, so a bad name, version or ``until`` is reported before a bad table or column.

        Preconditions:
            None.

        Returns:
            None.

        Raises:
            ValueError: From ``_check_declaration``, if ``name``, ``version``, ``severity`` or ``until`` breaks an
                invariant; or if ``table`` does not match ``_TABLE`` or ``column`` does not match ``_COLUMN``.

        """
        _check_declaration(self.name, self.version, self.severity, self.until)
        if not _TABLE.fullmatch(self.table):
            raise ValueError(f"lookup {self.name!r}: table must be schema.table in lower case, got {self.table!r}")
        if not _COLUMN.fullmatch(self.column):
            raise ValueError(f"lookup {self.name!r}: column must be a lower-case name, got {self.column!r}")


def _check_declaration(name: str, version: int, severity: Severity, until: str | None) -> None:
    """Check the fields shared by ``Rule`` and ``Lookup``.

    ``until`` is checked only for presence and for being non-empty; a whitespace-only value, or one that names no
    job, is accepted.

    Preconditions:
        None.

    Args:
        name: The check's name. Must be non-empty and not ``SCHEMA_RULE``.
        version: The check's version. Must be 1 or more.
        severity: The check's severity.
        until: The job a ``WAIT`` check waits for. Must be set exactly when ``severity`` is ``Severity.WAIT``, and
            must then not be ``""``.

    Returns:
        None.

    Raises:
        ValueError: If any rule above is broken. The first broken rule, in the order listed, is reported.

    """
    if not name or name == SCHEMA_RULE:
        raise ValueError(f"a check needs a name other than {SCHEMA_RULE!r}, got {name!r}")
    if version < 1:
        raise ValueError(f"check {name!r}: version must be 1 or more, got {version}")
    if (severity is Severity.WAIT) != (until is not None):
        raise ValueError(f"check {name!r}: until is set exactly when severity is wait")
    if until == "":
        raise ValueError(f"check {name!r}: until must name a job")


@dataclass(frozen=True, slots=True, kw_only=True)
class BatchPolicy:
    """The share and minimum count of held rows at which a batch fails. Frozen and keyword-only.

    Invariants, all enforced by ``__post_init__``:
        ``0 < share < 1``.
        ``min_rows`` is 1 or more.

    Attributes:
        share: The share of the batch that held rows must exceed for the batch to fail. Defaults to ``0.05``.
        min_rows: The fewest held rows that can fail a batch on share alone. Defaults to 3.

    """

    share: float = 0.05  # from 60 records up (with min_rows 3): fail when more than this share is held
    min_rows: int = 3  # below 60 records: fail when this many are held

    def __post_init__(self) -> None:
        """Reject a policy that breaks the class invariants.

        A ``NaN`` ``share`` is rejected, because every comparison with it is false.

        Preconditions:
            None.

        Returns:
            None.

        Raises:
            ValueError: If ``share`` is not strictly between 0 and 1, or ``min_rows`` is less than 1.

        """
        if not 0 < self.share < 1:
            raise ValueError(f"share must be between 0 and 1, got {self.share}")
        if self.min_rows < 1:
            raise ValueError(f"min_rows must be 1 or more, got {self.min_rows}")

    def fails(self, total: int, held: int) -> bool:
        """Return whether a batch with ``held`` of its ``total`` rows held fails.

        An empty batch never fails. Otherwise the batch fails when every row is held, whatever the count, or when at
        least ``min_rows`` rows are held and they are more than ``share * total``, compared exactly, taking ``share``
        as the decimal written (``Fraction(str(share))``), so 29 held of 100 with share 0.29 does not fail.

        Preconditions:
            ``0 <= held <= total``. Not checked: a negative ``total`` or a ``held`` above ``total`` gives a result
            but not a meaningful one.

        Args:
            total: The number of rows in the batch.
            held: The number of those rows that were held.

        Returns:
            True if the batch fails, False otherwise.

        Raises:
            Nothing.

        """
        if total == 0:
            return False
        share = Fraction(str(self.share))  # the decimal as written, so 0.29 of 100 is exactly 29
        return held == total or (held >= self.min_rows and held > share * total)


@dataclass(frozen=True, slots=True, kw_only=True)
class Validation[R: Record]:
    """A record model with its uniquely named checks and the batch policy that applies to them.

    Invariants, all enforced by ``__post_init__``:
        No two entries of ``checks`` share a ``name``; ``Rule`` and ``Lookup`` names share one namespace.

    Attributes:
        model: The record class the checks apply to. Stored only; not used on this page.
        checks: The rules and lookups, in declaration order. Defaults to none.
        policy: The policy that decides when the batch fails. Defaults to ``BatchPolicy()``.

    """

    model: type[R]
    checks: tuple[Rule[R] | Lookup[R], ...] = ()
    policy: BatchPolicy = field(default_factory=BatchPolicy)

    def __post_init__(self) -> None:
        """Reject a set of checks in which a name repeats.

        Preconditions:
            None.

        Returns:
            None.

        Raises:
            ValueError: If two or more checks share a name. The message lists every repeated name once, sorted,
                joined by ``", "``.

        """
        names = [check.name for check in self.checks]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(f"check names must be unique, repeated: {', '.join(repeated)}")
