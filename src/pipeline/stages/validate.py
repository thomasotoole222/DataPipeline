"""Validate the records a task landed in ``raw.page``: parse, deduplicate, run checks, and split accepted from held.

Public API:
    Validated: The outcome of ``validate``: accepted records, where they landed, held rows and counts.
    validate: Parse, deduplicate and check a task's landed records.
    held_accepted: Build the held row for an accepted record that a later stage holds back.
    judge: Fail the batch when its held rows break the batch policy.

"""

import json
import logging
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from pydantic import BaseModel, ValidationError

from pipeline.checks.checks import SCHEMA_RULE, BatchPolicy, CheckContext, Lookup, Rule, Severity, Validation
from pipeline.checks.record import Keyed, Record, key_of
from pipeline.connectors.spec import EndpointSpec
from pipeline.db import raw
from pipeline.db.control import ClaimedTask
from pipeline.db.lookup import known_values
from pipeline.db.pool import Pool
from pipeline.db.quarantine import Failure, HeldRow
from pipeline.db.raw import RawPage
from pipeline.errors import QualityFailed
from pipeline.telemetry import Instruments
from pipeline.timeutil import TimeFrame

#: Logger for this module; ``_report`` writes its warnings here.
_log: Final = logging.getLogger(__name__)
#: The most example keys ``_report`` puts in one log line.
_EXAMPLES: Final = 5


@dataclass(frozen=True, slots=True, kw_only=True)
class Validated[R: Record]:
    """The outcome of ``validate``: accepted records, where they landed, held rows and counts.

    Frozen, slotted and keyword-only. No field is checked; there is no ``__post_init__``.

    Attributes:
        accepted: Records kept by ``_dedupe`` that failed no non-``WARN`` check, in landed order, each with the key
            it was de-duplicated on, which Transform writes as its key columns.
        landed: Where each accepted record landed and its raw value, in ``accepted`` order, so ``held_accepted`` can
            hold it back after Validate.
        held: Rows held back, one per record that failed to parse, parsed with a key that could not be read, shares
            its key with a record that failed to parse, or failed a non-``WARN`` check.
        records: Number of records landed, before deduplication.
        deduplicated: Number of records dropped by ``_dedupe``.
        judged_total: The total ``judge`` was given: records landed, less those ``_dedupe`` dropped and the held
            copies of a failed key after the first.
        judged: The held rows ``judge`` was given; Load gives them to ``judge`` again with its own held records.

    """

    accepted: tuple[Keyed[R], ...]  # records that load with their keys, warned ones included, in fetch order
    landed: tuple[_Landed, ...]  # where each accepted record came from, in accepted order
    held: tuple[HeldRow, ...]  # records held back, for Load to write with quarantine.hold
    records: int  # records on the task's pages, duplicates included
    deduplicated: int  # earlier copies dropped because a later page had the same key
    judged_total: int  # the total judge was given: records landed, less dropped copies and repeats of failed keys
    judged: tuple[HeldRow, ...]  # the held rows judge was given; Load judges them again with its own


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
    index: int  # position in the page's records
    payload: Any  # the record as landed


@dataclass(frozen=True, slots=True)
class _Parsed[R: Record]:
    """A landed record together with the model instance parsed from it and its key.

    Frozen and slotted. No field is checked.

    Attributes:
        landed: Where the record came from and its raw value.
        record: The parsed record.
        key: The record's key, ``key_of`` of its model and ``landed.payload``.

    """

    landed: _Landed
    record: R
    key: tuple[object, ...]  # read from the payload, as for a copy that failed to parse


async def validate[R: Record](
    pool: Pool,
    task: ClaimedTask,
    spec: EndpointSpec,
    frame: TimeFrame | None,  # the task's time window, as Extract was given it
    validation: Validation[R],
    instruments: Instruments,
) -> Validated[R]:
    """Parse, deduplicate and check the records a task landed, and split them into accepted and held.

    Reads every page of ``task`` from ``raw.page`` on one pooled connection and flattens them into records with
    ``_records``. ``_parse_records`` parses them, holds every record that failed to parse and every parsed copy of a
    key that failed to parse, and returns the other parsed records. Then, once: ``_dedupe`` drops earlier copies of a
    key among those records, ``_fetch_lookups`` makes one ``known_values`` call per ``Lookup`` over the kept
    records, and each kept record is checked. Failures with severity ``WARN`` are only reported, with the record's
    key; any other failure (error or wait) holds the record, and a record with no such failure is accepted. Then
    ``_report`` and ``judge`` run, and the result is returned.

    A record that fails both an error check and a wait check is held with both failures and counts as an error,
    because ``_waiting`` needs every failure to have an ``until`` job. Every ``SCHEMA_RULE`` failure (a parse
    failure, an unreadable key, or a parsed copy of a key that failed to parse) counts as an ``ERROR`` failure in
    ``_report``.

    ``judge`` is given the rows ``_parse_records`` counts (one per failed key and one per record whose key could not
    be read) and every held kept record, and a total of the records landed less those ``_dedupe`` dropped and the held
    rows it does not count. So a key fetched three times counts as one record and one held row whether it fails to
    parse or fails an error check.

    Preconditions:
        None.

    Args:
        pool: Pool the pages and the lookups are read through.
        task: The claimed task whose pages are validated; gives the id, job, source and partition.
        spec: The endpoint spec that says where the records are in a page body.
        frame: The task's time frame, or ``None``; passed to checks through ``CheckContext``.
        validation: The record model and the checks to run.
        instruments: Telemetry passed to ``_parse_records`` and ``_report``.

    Returns:
        A ``Validated`` with the accepted records and their keys, and where each landed; every held row, parse
        failures and parsed records with an unreadable key first in landed order, then the parsed copies of
        failed keys in landed order, then check failures in kept order; ``records``, the number of records
        landed before deduplication; ``deduplicated``, the number of records dropped by ``_dedupe``; and
        ``judged_total`` and ``judged``, the total and the held rows ``judge`` was given.

    Raises:
        QualityFailed: From ``judge``, when the held rows that are not waiting, each failed key counted once, break
            the batch policy.
        ClientErrorUnhandled: From ``_records``, when a page other than the last has nothing or ``null`` at the
            records path, or any page has a non-list there.
        DatabaseConflict: From ``raw.read_pages`` or ``known_values`` through ``db_errors``.
        DatabaseUnavailable: From ``raw.read_pages`` or ``known_values`` through ``db_errors``.
        psycopg.Error: From ``raw.read_pages`` or ``known_values``, any other database error unchanged (e.g.
            ``UndefinedTable`` for a lookup table that does not exist).
        TypeError: From ``raw.read_pages``, when building a ``RawPage`` from a row whose payload is not an object;
            or from ``_lookup_values``, when a lookup's ``values`` returns a bare ``str``.
        ValueError: From ``raw.read_pages``, when building a ``RawPage`` from a row whose ``page_no`` is negative.
        Exception: Any exception a rule's predicate or a lookup's ``values`` raises propagates unchanged; so does
            any exception ``_parse`` lets through ``_parse_records``.

    """
    ctx = CheckContext(job=task.job, source_id=task.source_id, partition_key=task.partition_key, frame=frame)
    async with pool.connection() as conn:
        pages = [page async for page in raw.read_pages(conn, task.id)]
    landed = _records(pages, spec)
    clean, held, counted, new_fields = _parse_records(task, validation.model, landed, instruments)
    failed = [(Severity.ERROR, failure) for row in held for failure in row.failures]  # schema failures, held order
    kept, dropped = _dedupe(clean)
    known = await _fetch_lookups(pool, validation.checks, [p.record for p in kept])
    accepted: list[Keyed[R]] = []
    origins: list[_Landed] = []  # where each accepted record landed
    warned: list[tuple[Failure, tuple[object, ...]]] = []
    warned_rows = 0
    for entry in kept:
        failures = _check(entry.record, ctx, validation.checks, known)
        failed.extend(failures)
        kept_failures = tuple(failure for severity, failure in failures if severity is not Severity.WARN)
        warnings = [failure for severity, failure in failures if severity is Severity.WARN]
        warned.extend((failure, entry.key) for failure in warnings)
        if kept_failures:
            held.append(_held(task, entry.landed, _group_key(entry.key), kept_failures))
            counted.append(held[-1])
        else:
            accepted.append(Keyed(entry.record, entry.key))
            origins.append(entry.landed)
            warned_rows += bool(warnings)
    _report(task, instruments, held, warned, warned_rows, failed, dropped, new_fields)
    repeats = len(held) - len(counted)  # the other copies of failed keys, counted once like deduplicated ones
    judged_total = len(landed) - len(dropped) - repeats
    judge(validation.policy, judged_total, counted)
    return Validated(
        accepted=tuple(accepted),
        landed=tuple(origins),
        held=tuple(held),
        records=len(landed),
        deduplicated=len(dropped),
        judged_total=judged_total,
        judged=tuple(counted),
    )


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
            continue  # a last page may leave out its empty record list
        landed.extend(_Landed(page.page_no, index, item) for index, item in enumerate(spec.records(page.payload)))
    return landed


def _parse_records[R: Record](
    task: ClaimedTask, model: type[R], landed: Sequence[_Landed], instruments: Instruments
) -> tuple[list[_Parsed[R]], list[HeldRow], list[HeldRow], set[str]]:
    """Parse landed records, holding every record that failed to parse and every parsed copy of its key.

    Each landed record is parsed with ``_parse``: a parse failure is held at once, with its key read by ``_payload_key``
    and, as its group key, ``_group_key`` of the key of the first failed record whose key equals it (as values);
    otherwise its key is read from the same payload by ``_payload_key`` and, when readable, the parsed record is kept
    with it and its new fields are collected. When ``_payload_key`` cannot read a record's key, the record is held with
    no group key and nothing else is held for it: a parsed one with a ``SCHEMA_RULE`` failure carrying
    ``model.record_version`` and the message ``"the record's key could not be read"`` (e.g. a model validator reshaped
    the payload the key-only model reads). Once every record is parsed, if any key was unreadable, one warning naming
    the task and job and giving the count (never values) is logged, and ``instruments.rows`` gets the count with
    ``stage="key_unreadable"``. Every parsed record whose key equals (as values, not as text) the key of a record that
    failed to parse is then held under the first such failed record's group key, with a ``SCHEMA_RULE`` failure carrying
    ``model.record_version`` and the message ``"another copy of this record failed to parse"``.

    Every copy of a key that failed to parse, and every parsed copy of such a key, is its own held row, but only the
    first held row of each failed key is counted for the batch policy; a record whose key could not be read, parsed or
    not, is counted on its own.

    Preconditions:
        ``model`` is a subclass of ``Record``, as ``key_of`` requires.

    Args:
        task: The task that landed the records; gives the held rows' ids and the log and metric attributes.
        model: The record class to parse into.
        landed: The landed records, in landed order.
        instruments: ``rows`` is added to when a key is unreadable.

    Returns:
        ``(clean, held, counted, new_fields)``: the parsed records not held, in landed order; the held rows, parse
        failures and parsed records with an unreadable key in landed order, then parsed copies of failed keys in landed
        order, each with one ``SCHEMA_RULE`` failure; the held rows counted for the batch policy, one per failed key and
        one per unreadable key, in landed order; and the paths of undeclared fields found by ``_new_fields`` in every
        parsed record kept with its key.

    Raises:
        Exception: From ``_parse``, any exception other than ``ValidationError`` from ``model_validate``.

    """
    parsed: list[_Parsed[R]] = []
    held: list[HeldRow] = []
    counted: list[HeldRow] = []  # the held rows judged by the batch policy: one per failed key
    unparsed_keys: dict[tuple[object, ...], str] = {}  # a failed copy's key -> its group key
    new_fields: set[str] = set()
    unreadable = 0  # records whose key could not be read
    unkeyed = Failure(SCHEMA_RULE, model.record_version, "the record's key could not be read")
    for item in landed:
        outcome = _parse(model, item)
        if isinstance(outcome, Failure):
            key = _payload_key(model, item.payload)
            first = key is None or key not in unparsed_keys  # an unreadable key counts on its own
            if key is None:
                group_key = None
                unreadable += 1
            else:
                group_key = unparsed_keys.setdefault(key, _group_key(key))
            held.append(_held(task, item, group_key, (outcome,)))
            if first:
                counted.append(held[-1])
        elif (key := _payload_key(model, item.payload)) is None:  # parsed, but no key to load or dedupe it by
            unreadable += 1
            held.append(_held(task, item, None, (unkeyed,)))
            counted.append(held[-1])
        else:
            parsed.append(_Parsed(item, outcome, key))
            new_fields.update(_new_fields(outcome))
    if unreadable:  # one line per call, however many records
        _log.warning("task %s %s: %d records with an unreadable key", task.id, task.job, unreadable)
        instruments.rows.add(unreadable, {"job": task.job, "stage": "key_unreadable"})
    shadowed = Failure(SCHEMA_RULE, model.record_version, "another copy of this record failed to parse")
    clean: list[_Parsed[R]] = []
    for entry in parsed:
        group_key = unparsed_keys.get(entry.key)
        if group_key is not None:  # no copy may load while another copy of its key is held, earlier or later
            held.append(_held(task, entry.landed, group_key, (shadowed,)))
        else:
            clean.append(entry)
    return clean, held, counted, new_fields


def _parse[R: Record](model: type[R], item: _Landed) -> R | Failure:
    """Parse one landed record into ``model``, or describe why it does not parse.

    Calls ``model.model_validate(item.payload)``, in Python mode on the JSON-decoded payload; ``Record`` refuses a
    model whose config sets ``strict=True``, which would refuse the JSON string forms of dates, UUIDs and decimals.
    If that raises ``ValidationError``, the errors are joined as
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
        Exception: Any exception from ``model_validate`` other than ``ValidationError`` propagates unchanged, for
            example one raised inside a validator that is not a ``ValueError`` or ``AssertionError`` (Pydantic turns
            those two into ``ValidationError``). ``_parse`` raises nothing itself.

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
    with ``path + name + "."`` as the new path. For any ``Mapping`` (a dict included): the union over its values,
    with ``path`` minus a trailing ``"."`` plus ``"{}."`` as the new path; its keys are not searched. For any other
    ``Collection`` (sized and re-iterable) except ``str``, ``bytes`` and ``bytearray`` (a list, tuple, set, frozenset
    or ``deque`` included): the union over its items, with ``path`` minus a trailing ``"."`` plus ``"[]."`` as the
    new path. Any other value gives an empty set, so a one-shot iterator or generator is never consumed. For
    example, an extra ``zip`` inside field ``address`` gives ``"address.zip"``, an extra ``promo`` in a list field
    ``lines`` gives ``"lines[].promo"``, and in a dict field ``by_code`` ``"by_code{}.promo"``. Declared fields are
    named by their Python field name, not their alias; extra fields by the name they had in the payload.

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
    if isinstance(value, Mapping):
        return {found for item in value.values() for found in _new_fields(item, f"{path.removesuffix('.')}{{}}.")}
    if isinstance(value, Collection) and not isinstance(value, str | bytes | bytearray):
        return {found for item in value for found in _new_fields(item, f"{path.removesuffix('.')}[].")}
    return set()


def _dedupe[R: Record](parsed: Sequence[_Parsed[R]]) -> tuple[list[_Parsed[R]], list[_Parsed[R]]]:
    """Split parsed records into kept and dropped by key, by the position of each key's last occurrence.

    For each key (``_Parsed.key``), the record at its last position is kept: the copy fetched last, by later page
    and then later index. Every earlier copy is dropped, so with all keys distinct nothing is dropped.

    Preconditions:
        None.

    Args:
        parsed: The parsed records, in landed order.

    Returns:
        ``(kept, dropped)``, each in landed order.

    Raises:
        Nothing: ``Record`` allows only key types whose values are hashable.

    """
    last = {item.key: position for position, item in enumerate(parsed)}  # a later copy wins
    keep = set(last.values())
    kept = [item for position, item in enumerate(parsed) if position in keep]
    dropped = [item for position, item in enumerate(parsed) if position not in keep]
    return kept, dropped


async def _fetch_lookups[R: Record](
    pool: Pool, checks: Iterable[Rule[R] | Lookup[R]], records: Sequence[R]
) -> dict[str, frozenset[str]]:
    """Fetch, for each ``Lookup`` in ``checks``, which of the records' values exist in its table.

    Returns ``{}`` without a connection when there are no lookups or no records. Otherwise it opens one pooled
    connection and makes one ``known_values`` call per lookup, in check order, on the union of ``lookup.values``
    over all records.

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
        psycopg.Error: From ``known_values``, any other database error unchanged.
        TypeError: From ``_lookup_values``, when a lookup's ``values`` returns a bare ``str``.
        Exception: Any exception a lookup's ``values`` raises propagates unchanged.

    """
    lookups = [check for check in checks if isinstance(check, Lookup)]
    if not lookups or not records:
        return {}
    known: dict[str, frozenset[str]] = {}
    async with pool.connection() as conn:
        for lookup in lookups:
            wanted = {value for record in records for value in _lookup_values(lookup, record)}
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
        TypeError: From ``_lookup_values``, when a lookup's ``values`` returns a bare ``str``.
        Exception: Any exception from a rule's predicate or a lookup's ``values`` propagates; nothing else is raised.

    """
    failures: list[tuple[Severity, Failure]] = []
    for check in checks:
        if isinstance(check, Rule):
            if not check.check(record, ctx):
                failures.append((check.severity, Failure(check.name, check.version, "check failed", check.until)))
        else:
            missing = sorted(set(_lookup_values(check, record)) - known.get(check.name, frozenset()))
            if missing:
                message = f"not in {check.table}.{check.column}: {', '.join(missing)}"
                failures.append((check.severity, Failure(check.name, check.version, message, check.until)))
    return failures


def _lookup_values[R: Record](lookup: Lookup[R], record: R) -> Iterable[str]:
    """Return a lookup's values for one record, refusing a bare ``str``.

    ``Lookup.values`` is typed to return ``Iterable[str]``, which a ``str`` is, so a ``values`` that returns one value
    instead of a collection of them would otherwise be looked up one character at a time.

    Preconditions:
        None.

    Args:
        lookup: The lookup whose ``values`` is called.
        record: The record passed to it.

    Returns:
        What ``lookup.values(record)`` returned, unchanged.

    Raises:
        TypeError: ``lookup.values`` returned a ``str``.
        Exception: Any exception ``lookup.values`` raises propagates unchanged.

    """
    values = lookup.values(record)
    if isinstance(values, str):
        raise TypeError(f"lookup {lookup.name!r}: values must return an iterable of strings, not a str")
    return values


def _held(task: ClaimedTask, item: _Landed, group_key: str | None, failures: tuple[Failure, ...]) -> HeldRow:
    """Build the held row for one landed record from the task it came from and the checks it failed.

    A ``dict`` payload is stored as is, not copied; any other JSON value is wrapped as ``{"record": payload}``.

    Preconditions:
        ``failures`` is not empty, as ``HeldRow`` requires; ``_parse_records`` passes one schema failure and
        ``validate`` a non-empty ``kept_failures``. ``item.page_no`` and ``item.index`` are 0 or more, as ``RawPage``
        and ``enumerate`` give them.

    Args:
        task: The task that landed the record; gives the id, connector, job, source and partition.
        item: The landed record; gives its page, position and payload.
        group_key: The record's key from ``_group_key``, or ``None`` when ``_payload_key`` could not read its key,
            whether or not the record parsed.
        failures: The failures that hold it.

    Returns:
        The ``HeldRow``.

    Raises:
        ValueError: From ``HeldRow``, if ``failures`` is empty.

    """
    payload = item.payload if isinstance(item.payload, dict) else {"record": item.payload}
    return HeldRow(
        task_id=task.id,
        connector=task.connector,
        job=task.job,
        source_id=task.source_id,
        partition_key=task.partition_key,
        raw_page_no=item.page_no,
        raw_index=item.index,
        group_key=group_key,
        payload=payload,
        failures=failures,
    )


def held_accepted(task: ClaimedTask, validated: Validated[Any], position: int, failure: Failure) -> HeldRow:
    """Build the held row for an accepted record that a later stage holds back, such as Transform or Load.

    The row is built as ``validate`` builds one for a failed check: by ``_held``, from the record's landed page,
    position and payload, with ``_group_key`` of its key and ``failure`` as its one failure.

    Preconditions:
        ``validated`` is ``validate``'s result for ``task``.

    Args:
        task: The task that landed the record; gives the id, connector, job, source and partition.
        validated: The validate result holding the record.
        position: The record's position in ``validated.accepted``.
        failure: Why the record is held.

    Returns:
        The ``HeldRow``.

    Raises:
        IndexError: When ``position`` is not a position in ``validated.accepted``.

    """
    return _held(task, validated.landed[position], _group_key(validated.accepted[position].key), (failure,))


def _group_key(key: Iterable[object]) -> str:
    """Return the group key for a record's key values: a compact JSON array of the values as strings.

    For example ``("a|b", "c")`` gives ``'["a|b","c"]'``. Two keys give the same group key exactly when their values,
    as ``str``, are equal in order.

    Preconditions:
        None.

    Args:
        key: The key values in ``record_key`` order, from ``key_of`` (directly or through ``_payload_key``).

    Returns:
        ``json.dumps([str(value) for value in key], separators=(",", ":"))``.

    Raises:
        Nothing.

    """
    return json.dumps([str(value) for value in key], separators=(",", ":"))


def _payload_key(model: type[Record], payload: Any) -> tuple[object, ...] | None:
    """Return the key of a landed record, parsed or not, read by ``key_of``, or ``None`` when it is unreadable.

    ``key_of`` reads every key, a parsed record's included, from the payload with the same key-only model, so the key
    read here equals the key of a parsed copy of the same key. Any exception while reading the key makes it
    unreadable.

    Preconditions:
        ``model`` is a subclass of ``Record``, as ``key_of`` requires.

    Args:
        model: The record class the payload was parsed into.
        payload: The record as landed; any JSON value.

    Returns:
        ``key_of(model, payload)``; ``None`` when it raises (e.g. ``payload`` is not an object, or a key field is
        absent, ``null`` or of the wrong type).

    Raises:
        Nothing: every ``Exception`` from ``key_of``, ``ValidationError`` included, is caught.

    """
    try:
        return key_of(model, payload)
    except Exception:  # the key is unreadable, whatever was raised; the record is held without it
        return None


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


def judge(policy: BatchPolicy, total: int, held: Sequence[HeldRow]) -> None:
    """Fail the batch when the held rows that are not waiting break the batch policy.

    Rows for which ``_waiting`` is true are not counted. When ``policy.fails(total, errors)`` is true it raises
    ``QualityFailed(rows_total=total, rows_quarantined=<non-waiting held rows>, by_rule=<failures per rule over those
    rows>)``; a row that broke two rules counts under both.

    Preconditions:
        ``0 <= len(held) <= total``, as ``BatchPolicy.fails`` expects; ``validate`` guarantees this because every row
        it passes is the first copy of a failed key, a parse failure with an unreadable key, or a kept record, and
        ``total`` counts each of them once. ``stages.load`` adds only accepted records, each a kept record that
        ``total`` counts and ``validate`` did not hold.

    Args:
        policy: The batch policy.
        total: Records after deduplication, each failed key counted once, parse failures included.
        held: The task's held rows to judge: one per failed key, every one whose key is unreadable, and every held
            kept record; ``stages.load`` passes ``Validated.judged`` followed by the records it holds.

    Returns:
        None.

    Raises:
        QualityFailed: When the policy fails the batch.

    """
    errors = [row for row in held if not _waiting(row)]
    if policy.fails(total, len(errors)):
        by_rule = Counter(failure.rule for row in errors for failure in row.failures)
        raise QualityFailed(rows_total=total, rows_quarantined=len(errors), by_rule=by_rule)


def _report(
    task: ClaimedTask,
    instruments: Instruments,
    held: Sequence[HeldRow],
    warned: Sequence[tuple[Failure, tuple[object, ...]]],
    warned_rows: int,
    failed: Sequence[tuple[Severity, Failure]],
    dropped: Sequence[_Parsed[Any]],
    new_fields: set[str],
) -> None:
    """Count and log the task's held, warned and deduplicated records, its failures and its new fields.

    ``instruments.rows`` is added to once per stage, with ``job`` and ``stage`` attributes: a held row under
    ``"waiting"`` when ``_waiting`` is true for it, otherwise under ``"quarantined"``; ``warned_rows`` under
    ``"warned"``; and each dropped record under ``"deduplicated"``. A stage with no records is not added to. A record
    whose key could not be read is also counted under ``"key_unreadable"`` by ``_parse_records``, so such a record
    is counted under two stages.
    ``instruments.check_failures`` counts each failure in ``failed`` once, with ``job``, ``rule`` and ``severity``
    (the ``Severity`` value) attributes, in sorted ``(rule, severity)`` order.

    For the log, held rows and warnings are grouped by ``(stage, rule, version)``, with ``stage`` as above for a held
    row and ``"warned"`` for a warning. A held row adds one entry per failure, and every ``WARN`` failure adds one,
    whether its record was accepted or held. Held rows give their group key, or ``"page <n> #<i>"`` when it is
    ``None`` or ``""``; warnings give the record's key as ``_group_key`` spells it. For each group, in sorted order,
    it logs one warning with the count of entries and up to ``_EXAMPLES`` distinct example keys, in first-seen order.
    Deduplicated records are logged with their count and up to ``_EXAMPLES`` distinct keys, each from ``_group_key``;
    new fields are logged once, sorted, joined with ``", "``.

    Preconditions:
        None.

    Args:
        task: Gives the id and job for the metric attributes and log lines.
        instruments: ``rows`` and ``check_failures`` are the counters added to.
        held: Every held row of the task.
        warned: Each ``WARN`` failure with the key of the record it failed on.
        warned_rows: The number of accepted records with at least one ``WARN`` failure.
        failed: Every failure of the task with its severity: schema failures as ``ERROR``, then each check failure.
        dropped: The records ``_dedupe`` dropped.
        new_fields: Paths of undeclared fields found by ``_new_fields``.

    Returns:
        None.

    Raises:
        Nothing.

    """
    stages = Counter("waiting" if _waiting(row) else "quarantined" for row in held)
    stages["warned"] = warned_rows
    stages["deduplicated"] = len(dropped)
    for stage, count in sorted(stages.items()):
        if count:
            instruments.rows.add(count, {"job": task.job, "stage": stage})
    for (rule, severity), count in sorted(Counter((failure.rule, severity) for severity, failure in failed).items()):
        instruments.check_failures.add(count, {"job": task.job, "rule": rule, "severity": severity.value})
    examples: dict[tuple[str, str, int], list[str]] = {}
    for row in held:
        stage = "waiting" if _waiting(row) else "quarantined"
        for failure in row.failures:
            examples.setdefault((stage, failure.rule, failure.version), []).append(
                row.group_key or f"page {row.raw_page_no} #{row.raw_index}"
            )
    for failure, key in warned:
        examples.setdefault(("warned", failure.rule, failure.version), []).append(_group_key(key))
    for (stage, rule, version), keys in sorted(examples.items()):
        _log.warning(
            "task %s %s: %d %s by %s v%d, e.g. %s",
            task.id,
            task.job,
            len(keys),
            stage,
            rule,
            version,
            ", ".join(list(dict.fromkeys(keys))[:_EXAMPLES]),  # distinct, so repeats cannot crowd others out
        )
    if dropped:
        distinct = dict.fromkeys(_group_key(item.key) for item in dropped)  # in first-seen order
        sample = ", ".join(list(distinct)[:_EXAMPLES])
        _log.warning("task %s %s: %d deduplicated, e.g. %s", task.id, task.job, len(dropped), sample)
    if new_fields:
        _log.warning("task %s %s: new fields %s", task.id, task.job, ", ".join(sorted(new_fields)))
