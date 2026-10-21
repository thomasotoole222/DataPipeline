"""Validate the records a task landed in ``raw.page``: parse, deduplicate, run checks, and split accepted from held.

Public API:
    Validated: The outcome of ``validate``: accepted records, held rows and counts.
    validate: Parse, deduplicate and check a task's landed records.

"""

import logging
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from pydantic import BaseModel, ValidationError

from cipher_pipeline.checks.checks import SCHEMA_RULE, CheckContext, Lookup, Rule, Severity, Validation
from cipher_pipeline.checks.record import Record, key_of
from cipher_pipeline.db import raw
from cipher_pipeline.db.control import ClaimedTask
from cipher_pipeline.db.lookup import known_values
from cipher_pipeline.db.pool import Pool
from cipher_pipeline.db.quarantine import Failure, HeldRow
from cipher_pipeline.db.raw import RawPage
from cipher_pipeline.errors import QualityFailed
from cipher_pipeline.sources.spec import EndpointSpec
from cipher_pipeline.telemetry import Instruments
from cipher_pipeline.timeutil import TimeFrame

#: Logger for this module; ``_report`` writes its warnings here.
_log: Final = logging.getLogger(__name__)
#: The most example keys ``_report`` puts in one log line.
_EXAMPLES: Final = 5


@dataclass(frozen=True, slots=True, kw_only=True)
class Validated[R: Record]:
    """The outcome of ``validate``: accepted records, held rows and counts.

    Frozen, slotted and keyword-only. No field is checked; there is no ``__post_init__``.

    Attributes:
        accepted: Records that passed every non-``WARN`` check.
        held: Rows held back, one per record that failed to parse or failed a non-``WARN`` check.
        records: Number of records landed, before deduplication.
        deduplicated: Number of records dropped by ``_dedupe``.

    """

    accepted: tuple[R, ...]
    held: tuple[HeldRow, ...]
    records: int
    deduplicated: int


@dataclass(frozen=True, slots=True)
class _Landed:
    """One record as landed: its page, its position in the page, and its raw JSON value.

    Frozen and slotted. No field is checked.

    Attributes:
        page_no: ``page_no`` of the ``RawPage`` the record came from.
        index: Position of the record in that page's records list.
        payload: The record as decoded from JSON; any type.

    """

    page_no: int
    index: int
    payload: Any


@dataclass(frozen=True, slots=True)
class _Parsed[R: Record]:
    """A landed record together with the model instance parsed from it.

    Frozen and slotted. No field is checked.

    Attributes:
        landed: Where the record came from and its raw value.
        record: The parsed record.

    """

    landed: _Landed
    record: R


async def validate[R: Record](
    pool: Pool,
    task: ClaimedTask,
    spec: EndpointSpec,
    frame: TimeFrame | None,
    validation: Validation[R],
    instruments: Instruments,
) -> Validated[R]:
    """Parse, deduplicate and check the records a task landed, and split them into accepted and held.

    Reads every page of ``task`` from ``raw.page`` on one pooled connection and flattens them into records with
    ``_records``. Each landed record is parsed: a parse failure is held at once with no group key; otherwise the
    parsed record is kept and its new fields are collected. Then, once: ``_dedupe`` drops earlier copies of a key, one
    lookup query per ``Lookup`` runs over the kept records, and each kept record is checked. Failures with severity
    ``WARN`` are only reported, with the record's key; any other failure (error or wait) holds the record, and a
    record with no such failure is accepted. Then ``_report`` and ``_judge`` run, and the result is returned.

    A record that fails both an error check and a wait check is held with both failures and counts as an error,
    because ``_waiting`` needs every failure to have an ``until`` job.

    Preconditions:
        None.

    Args:
        pool: Pool the pages and the lookups are read through.
        task: The claimed task whose pages are validated; gives the id, job, shop and partition.
        spec: The endpoint spec that says where the records are in a page body.
        frame: The task's time frame, or ``None``; passed to checks through ``CheckContext``.
        validation: The record model and the checks to run.
        instruments: Telemetry passed to ``_report``.

    Returns:
        A ``Validated`` with the accepted records; the held rows, parse failures first in landed order and then check
        failures in kept order; ``records``, the number of records landed before deduplication; and
        ``deduplicated``, the number of records dropped by ``_dedupe``.

    Raises:
        QualityFailed: From ``_judge``, when the held rows that are not waiting break the batch policy.
        ClientErrorUnhandled: From ``_records``, when a page other than the last has no records list.
        DatabaseConflict: From ``raw.read_pages`` or ``known_values`` through ``db_errors``.
        DatabaseUnavailable: From ``raw.read_pages`` or ``known_values`` through ``db_errors``.
        Exception: Any exception a rule's predicate or a lookup's ``values`` raises propagates unchanged.

    """
    ctx = CheckContext(job=task.job, shop_id=task.shop_id, partition_key=task.partition_key, frame=frame)
    async with pool.connection() as conn:
        pages = [page async for page in raw.read_pages(conn, task.id)]
    landed = _records(pages, spec)
    parsed: list[_Parsed[R]] = []
    held: list[HeldRow] = []
    new_fields: set[str] = set()
    for item in landed:
        outcome = _parse(validation.model, item)
        if isinstance(outcome, Failure):
            held.append(_held(task, item, None, (outcome,)))
        else:
            parsed.append(_Parsed(item, outcome))
            new_fields.update(_new_fields(outcome))
    kept, dropped = _dedupe(parsed)
    known = await _fetch_lookups(pool, validation.checks, [p.record for p in kept])
    accepted: list[R] = []
    warned: list[tuple[Failure, tuple[object, ...]]] = []
    for entry in kept:
        failures = _check(entry.record, ctx, validation.checks, known)
        kept_failures = tuple(failure for severity, failure in failures if severity is not Severity.WARN)
        warned.extend((failure, key_of(entry.record)) for severity, failure in failures if severity is Severity.WARN)
        if kept_failures:
            held.append(_held(task, entry.landed, _group_key(entry.record), kept_failures))
        else:
            accepted.append(entry.record)
    _report(task, instruments, held, warned, dropped, new_fields)
    _judge(validation, len(landed) - len(dropped), held)
    return Validated(accepted=tuple(accepted), held=tuple(held), records=len(landed), deduplicated=len(dropped))


def _records(pages: Iterable[RawPage], spec: EndpointSpec) -> list[_Landed]:
    """Flatten pages into landed records, in page order and then record order.

    A page is skipped only when it is the last (``cursor_next is None``) and ``spec.record_count`` gives 0 for it, which
    covers a last page with no records list, a ``null`` one, or an empty one.

    Preconditions:
        None.

    Args:
        pages: The pages, in the order their records should come out.
        spec: Where the records list sits in a page body.

    Returns:
        One ``_Landed`` per element of each kept page's records list, with the page's ``page_no`` and the element's
        index.

    Raises:
        ClientErrorUnhandled: From ``spec.record_count`` or ``spec.records``, when a page that is not the last has
            nothing or ``null`` at the records path, or any page has a non-list there.

    """
    landed: list[_Landed] = []
    for page in pages:
        last = page.cursor_next is None
        if last and spec.record_count(page.payload, last_page=True) == 0:
            continue
        landed.extend(_Landed(page.page_no, index, item) for index, item in enumerate(spec.records(page.payload)))
    return landed


def _parse[R: Record](model: type[R], item: _Landed) -> R | Failure:
    """Parse one landed record into ``model``, or describe why it does not parse.

    Calls ``model.model_validate(item.payload)``. If that raises ``ValidationError``, the errors are joined as
    ``"loc: msg"`` with ``"; "``, using ``"(record)"`` for an empty location, into a ``SCHEMA_RULE`` failure carrying
    ``model.record_version``.

    Preconditions:
        None.

    Args:
        model: The record class to parse into.
        item: The landed record.

    Returns:
        The parsed record, or a ``SCHEMA_RULE`` ``Failure`` carrying ``model.record_version`` when pydantic raises
        ``ValidationError``.

    Raises:
        Nothing: ``_parse`` raises nothing itself, and ``ValidationError`` is caught. Any other exception from
            ``model_validate`` (for example one raised inside a validator that is not a ``ValueError`` or
            ``AssertionError``) propagates.

    """
    try:
        return model.model_validate(item.payload)
    except ValidationError as err:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or '(record)'}: {error['msg']}" for error in err.errors()
        )
        return Failure(SCHEMA_RULE, model.record_version, problems)


def _new_fields(value: object, path: str = "") -> set[str]:
    """Return the dotted paths of extra (undeclared) fields found in a parsed value.

    For a model: the names in ``model_extra``, prefixed by ``path``, plus the result for each declared field's value
    with ``path + name + "."`` as the new path. For a list or tuple: the union over its items, with ``path`` minus a
    trailing ``"."`` plus ``"[]."`` as the new path. Any other value gives an empty set. For example, an extra
    ``zip`` inside field ``address`` gives ``"address.zip"``, and an extra ``promo`` in a list field ``lines`` gives
    ``"lines[].promo"``.

    Preconditions:
        None.

    Args:
        value: The value to search.
        path: Prefix for found names; ``""`` at the top level.

    Returns:
        The set of paths found; empty when there are none.

    Raises:
        Nothing.

    """
    if isinstance(value, BaseModel):
        found = {f"{path}{name}" for name in value.model_extra or {}}
        for name in type(value).model_fields:
            found |= _new_fields(getattr(value, name), f"{path}{name}.")
        return found
    if isinstance(value, list | tuple):
        return {found for item in value for found in _new_fields(item, f"{path.removesuffix('.')}[].")}
    return set()


def _dedupe[R: Record](parsed: Sequence[_Parsed[R]]) -> tuple[list[_Parsed[R]], list[_Parsed[R]]]:
    """Split parsed records into kept and dropped by key, by the position of each key's last occurrence.

    For each ``key_of`` value, the record at its last position is kept: the copy fetched last, by later page and then
    later index. Every earlier copy is dropped, so with all keys distinct nothing is dropped.

    Preconditions:
        Every ``record`` is an instance of a ``Record`` subclass, as ``key_of`` requires.

    Args:
        parsed: The parsed records, in landed order.

    Returns:
        ``(kept, dropped)``, each in landed order.

    Raises:
        TypeError: From building the dict, when a key value is unhashable.

    """
    last = {key_of(item.record): position for position, item in enumerate(parsed)}  # a later copy wins
    keep = set(last.values())
    kept = [item for position, item in enumerate(parsed) if position in keep]
    dropped = [item for position, item in enumerate(parsed) if position not in keep]
    return kept, dropped


async def _fetch_lookups[R: Record](
    pool: Pool, checks: Iterable[Rule[R] | Lookup[R]], records: Sequence[R]
) -> dict[str, frozenset[str]]:
    """Fetch, for each ``Lookup`` in ``checks``, which of the records' values exist in its table.

    Returns ``{}`` without a connection when there are no lookups or no records. Otherwise it opens one pooled
    connection and runs one ``known_values`` query per lookup, in check order, on the union of ``lookup.values`` over
    all records.

    Preconditions:
        Lookup names are unique, as ``Validation.__post_init__`` guarantees.

    Args:
        pool: Pool to take the connection from.
        checks: The rules and lookups; rules are ignored.
        records: The records whose values are looked up.

    Returns:
        Lookup name to the subset of its wanted values that exist, as text.

    Raises:
        DatabaseConflict: From ``known_values``.
        DatabaseUnavailable: From ``known_values``.

    """
    lookups = [check for check in checks if isinstance(check, Lookup)]
    if not lookups or not records:
        return {}
    known: dict[str, frozenset[str]] = {}
    async with pool.connection() as conn:
        for lookup in lookups:
            wanted = {value for record in records for value in lookup.values(record)}
            known[lookup.name] = await known_values(conn, lookup.table, lookup.column, wanted)
    return known


def _check[R: Record](
    record: R, ctx: CheckContext, checks: Iterable[Rule[R] | Lookup[R]], known: Mapping[str, frozenset[str]]
) -> list[tuple[Severity, Failure]]:
    """Run every check on one record and return the failures with their severities.

    The type test is ``isinstance(check, Rule)``. A rule fails when its predicate returns false; a lookup fails when
    any of its values is missing from ``known[check.name]`` (an absent name counts as empty), with message
    ``"not in <table>.<column>: <missing values>"``. Each failure is built as ``Failure(name, version, message,
    until)`` with the check's ``until``.

    Preconditions:
        None.

    Args:
        record: The record to check.
        ctx: The context passed to each rule's predicate.
        checks: The rules and lookups, run in order.
        known: Lookup name to the values found for it, from ``_fetch_lookups``.

    Returns:
        One ``(severity, failure)`` per failed check, in check order; empty when ``checks`` is empty.

    Raises:
        Exception: Any exception from a rule's predicate or a lookup's ``values`` propagates; nothing else is raised.

    """
    failures: list[tuple[Severity, Failure]] = []
    for check in checks:
        if isinstance(check, Rule):
            if not check.check(record, ctx):
                failures.append((check.severity, Failure(check.name, check.version, "check failed", check.until)))
        else:
            missing = sorted(set(check.values(record)) - known.get(check.name, frozenset()))
            if missing:
                message = f"not in {check.table}.{check.column}: {','.join(missing)}"
                failures.append((check.severity, Failure(check.name, check.version, message, check.until)))
    return failures


def _held(task: ClaimedTask, item: _Landed, group_key: str | None, failures: tuple[Failure, ...]) -> HeldRow:
    """Build the held row for one landed record from the task it came from and the checks it failed.

    A ``dict`` payload is stored as is, not copied; any other JSON value is wrapped as ``{"record": payload}``.

    Preconditions:
        ``failures`` is not empty, as ``HeldRow`` requires; both callers in ``validate`` pass one parse failure or a
        non-empty ``kept_failures``. ``item.page_no`` and ``item.index`` are 0 or more, as ``RawPage`` and
        ``enumerate`` give them.

    Args:
        task: The task that landed the record; gives the id, job, shop and partition.
        item: The landed record; gives its page, position and payload.
        group_key: The record's key from ``_group_key``, or ``None`` when it did not parse.
        failures: The failures that hold it.

    Returns:
        The ``HeldRow``.

    Raises:
        ValueError: From ``HeldRow``, if ``failures`` is empty.

    """
    payload = item.payload if isinstance((item.payload), dict) else {"record": item.payload}
    return HeldRow(
        task_id=task.id,
        job=task.job,
        shop_id=task.shop_id,
        partition_key=task.partition_key,
        raw_page_no=item.page_no,
        raw_index=item.index,
        group_key=group_key,
        payload=payload,
        failures=failures,
    )


def _group_key(record: Record) -> str:
    """Return a record's key values as text joined by ``"|"``.

    A ``"|"`` inside a value is not escaped, so ``("a|b",)`` and ``("a", "b")`` give the same text.

    Preconditions:
        ``type(record)`` is a subclass of ``Record``, as ``key_of`` requires.

    Args:
        record: The parsed record.

    Returns:
        ``str`` of each key value in ``record_key`` order, joined by ``"|"``.

    Raises:
        AttributeError: From ``key_of``, under the conditions its contract gives.

    """
    return "|".join(str(part) for part in key_of(record))


def _waiting(row: HeldRow) -> bool:
    """Return whether every failure of a held row has an ``until`` job, i.e. the row waits instead of erring.

    Preconditions:
        None.

    Args:
        row: The held row.

    Returns:
        Whether every failure's ``until`` is not ``None``.

    Raises:
        Nothing.

    """
    return all(failure.until is not None for failure in row.failures)


def _judge(validation: Validation[Any], total: int, held: Sequence[HeldRow]) -> None:
    """Fail the batch when the held rows that are not waiting break the validation's batch policy.

    Rows for which ``_waiting`` is true are not counted. When ``policy.fails(total, errors)`` is true it raises
    ``QualityFailed(rows_total=total, rows_quarantined=<non-waiting held rows>, by_rule=<failures per rule over those
    rows>)``; a row that broke two rules counts under both.

    Preconditions:
        ``0 <= len(held) <= total``, as ``BatchPolicy.fails`` expects; ``validate`` guarantees this because every held
        row is a parse failure or a kept record.

    Args:
        validation: Gives the batch policy.
        total: Records after deduplication, parse failures included.
        held: Every held row of the task.

    Returns:
        None.

    Raises:
        QualityFailed: When the policy fails the batch.

    """
    errors = [row for row in held if not _waiting(row)]
    if validation.policy.fails(total, len(errors)):
        by_rule = Counter(failure.rule for row in errors for failure in row.failures)
        raise QualityFailed(rows_total=total, rows_quarantined=len(errors), by_rule=by_rule)


def _report(
    task: ClaimedTask,
    instruments: Instruments,
    held: Sequence[HeldRow],
    warned: Sequence[tuple[Failure, tuple[object, ...]]],
    dropped: Sequence[_Parsed[Any]],
    new_fields: set[str],
) -> None:
    """Count and log the task's held, warned and deduplicated records and its new fields.

    Held rows and warnings are grouped by ``(stage, rule, version)``, with ``stage`` ``"waiting"`` (a held row whose
    failures all have ``until``), ``"quarantined"`` (any other held row) or ``"warned"``. Held rows give their group
    key, or ``"page <n> # <i>"`` when they have none; warnings give the record's key joined with ``"|"``. For each
    group, in sorted order, it adds the group's size to ``instruments.rows`` with ``job``, ``stage`` and ``rule``
    attributes, and logs one warning with up to ``_EXAMPLES`` example keys. Deduplicated records are counted under
    stage ``"deduplicated"`` and logged with up to ``_EXAMPLES`` keys; new fields are logged once, sorted.

    Preconditions:
        None.

    Args:
        task: Gives the id and job for the metric attributes and log lines.
        instruments: ``rows`` is the counter added to.
        held: Every held row of the task.
        warned: Each ``WARN`` failure with the key of the record it failed on.
        dropped: The records ``_dedupe`` dropped.
        new_fields: Paths of undeclared fields found by ``_new_fields``.

    Returns:
        None.

    Raises:
        Nothing.

    """
    examples: dict[tuple[str, str, int], list[str]] = {}
    for row in held:
        stage = "waiting" if _waiting(row) else "quarantined"
        for failure in row.failures:
            examples.setdefault((stage, failure.rule, failure.version), []).append(
                row.group_key or f"page {row.raw_page_no} # {row.raw_index}"
            )
    for failure, key in warned:
        examples.setdefault(("warned", failure.rule, failure.version), []).append("|".join(map(str, key)))
    for (stage, rule, version), keys in sorted(examples.items()):
        instruments.rows.add(len(keys), {"job": task.job, "stage": stage, "rule": rule})
        _log.warning(
            "task %s %s: %d %s by %s v%d, e.g. %s",
            task.id,
            task.job,
            len(keys),
            stage,
            rule,
            version,
            ", ".join(keys[:_EXAMPLES]),
        )
    if dropped:
        instruments.rows.add(len(dropped), {"job": task.job, "stage": "deduplicated"})
        sample = ", ".join(_group_key(item.record) for item in dropped[:_EXAMPLES])
        _log.warning("task %s %s: %d deduplicated, e.g. %s", task.id, task.job, len(dropped), sample)
    if new_fields:
        _log.warning("task %s %s: new fields: %s", task.id, task.job, ", ".join(sorted(new_fields)))
