"""Declarations of where a fanned-out job's ids come from, and how an id is read.

A fanned-out job calls its endpoint once per id, or once per batch of ids, where its ``EndpointSpec.id_param`` puts
them. Nothing here fetches or queries: ``stages.extract`` runs the id source and the calls, and ``stages.validate``
reads the list rows back.

Public API:
    QUERY_PARAMS: The named parameters bound into a ``FromQuery`` statement.
    DEFAULT_CONCURRENCY: The default cap on a task's fanned-out calls in flight at once, 5.
    id_text: Return a JSON or SQL value as an id's text, or ``None`` when it cannot be an id.
    FromList: Ids read from the records of a list endpoint the task drains first.
    FromQuery: Ids from a read-only warehouse query run before the task fetches.
    IdSource: A job's id source, ``FromList`` or ``FromQuery``.

"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from pipeline.connectors.spec import EndpointSpec, KeyPath, check_key_path, value_at

#: The named parameters ``stages.extract`` binds into a ``FromQuery`` statement, written ``%(name)s``:
#: ``source_id`` (uuid), the task's source; ``partition_key`` (text), the task's partition key; and ``frame_start``
#: and ``frame_end`` (timestamptz), the task's time frame, half-open, both ``NULL`` for a ``"none"`` partition.
QUERY_PARAMS: Final = frozenset({"source_id", "partition_key", "frame_start", "frame_end"})
#: The default cap on a task's fanned-out calls in flight at once (``BatchJob.concurrency``).
DEFAULT_CONCURRENCY: Final = 5
#: Every ``%`` sequence psycopg reads in a query, by its grammar: ``%%``, ``%(<anything but ")">)<char>`` or
#: ``%<char>``, the character missing at the end of the query. Group ``name`` is a named placeholder's name, group
#: ``format`` its format character.
_PERCENT: Final = re.compile(r"%(?:%|\((?P<name>[^)]*)\)(?P<format>.?)|.?)", re.DOTALL)


def id_text(value: object) -> str | None:
    """Return a value as an id's text: a non-empty ``str`` as it is, an ``int`` in decimal.

    Preconditions:
        None.

    Args:
        value: A JSON value from a list record, or a value from a query row.

    Returns:
        The id's text, or ``None`` for any other value: ``None``, ``""``, a ``bool``, a float, an object or a list.

    Raises:
        Nothing.

    """
    if isinstance(value, str):
        return value or None
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return None


@dataclass(frozen=True, slots=True, kw_only=True)
class FromList:
    """Ids read from each record of a list endpoint, which the task drains, paged and windowed as usual, first.

    Frozen, slotted and keyword-only. Construction guarantees that ``id_at`` is a valid key path and that ``spec`` has
    no ``id_param``, so a list is never itself fanned out (no chaining). Nothing hashes one; ``hash()`` raises
    ``TypeError``.

    Attributes:
        spec (EndpointSpec): The list endpoint.
        id_at (KeyPath): Where each list record holds its id.
        keep_list_row (bool): Whether an id whose call returns no record loads its list row as the record instead,
            for an endpoint that leaves some items out (TikTok Get Product omits ``FREEZE`` and ``DELETED`` products).
            The job's one record model then reads both shapes, so a field only the per-id call returns must be
            optional. Needs one id per call, which ``BatchJob`` checks, not this class. Defaults to ``False``.

    """

    spec: EndpointSpec
    id_at: KeyPath  # e.g. ("id",) in each product of a product search
    keep_list_row: bool = False  # True: an id the per-id call returns nothing for loads its list row

    def __post_init__(self) -> None:
        """Reject a bad ``id_at`` and a list endpoint that is itself fanned out.

        Preconditions:
            None.

        Returns:
            None.

        Raises:
            ValueError: From ``check_key_path`` for a bad ``id_at``; or, naming the spec, when ``spec.id_param`` is
                set.

        """
        check_key_path(self.id_at)
        if self.spec.id_param is not None:
            raise ValueError(f"{self.spec.name}: a list endpoint takes no ids, so a fan-out cannot chain")

    def id_of(self, record: object) -> str:
        """Return a list record's id, read at ``id_at`` and written as ``id_text`` writes it.

        Preconditions:
            None.

        Args:
            record: One record from a list response; any JSON value.

        Returns:
            The id's text, never empty.

        Raises:
            ClientErrorUnhandled: From ``spec.unhandled`` when ``record`` is not an object, nothing is at ``id_at``,
                or the value there is not a non-empty string or an integer.

        """
        where = ".".join(self.id_at)
        if not isinstance(record, Mapping):
            raise self.spec.unhandled(f"a list record is a {type(record).__name__}, not an object")
        try:
            found: Any = value_at(record, self.id_at)
        except KeyError:
            raise self.spec.unhandled(f"a list record has no id at {where}") from None
        text = id_text(found)
        if text is None:
            raise self.spec.unhandled(f"a list record's id at {where} is not a string or an integer")
        return text


@dataclass(frozen=True, slots=True)
class FromQuery:
    """Ids from a SQL query run read-only on the warehouse pool before the task fetches, such as a shop's LIVE ids.

    The query returns one column, each row one id: a non-empty ``text`` or an integer. It takes only the named
    parameters in ``QUERY_PARAMS``, written ``%(name)s``; a literal ``%`` is written ``%%``. Frozen and slotted.
    Construction guarantees that ``sql`` is not blank and that every ``%`` sequence psycopg reads in it is ``%%`` or
    ``%(name)s`` with ``name`` in ``QUERY_PARAMS``, so a positional ``%s``, a lone ``%`` (``LIKE 'a%'``), another
    format such as ``%(source_id)d`` and an unknown name are refused when the job is declared, not on every run.

    Attributes:
        sql (str): The query, for example ``SELECT live_id FROM warehouse.live WHERE source_id = %(source_id)s AND
            started_at >= %(frame_start)s AND started_at < %(frame_end)s``.

    """

    sql: str

    def __post_init__(self) -> None:
        """Reject a blank query and any ``%`` sequence other than ``%%`` and ``%(name)s`` of a ``QUERY_PARAMS`` name.

        The sequences are read by psycopg's grammar (``_PERCENT``), wherever they are, in string literals and comments
        too, as psycopg reads them.

        Preconditions:
            None.

        Returns:
            None.

        Raises:
            ValueError: When ``sql`` is blank or only whitespace; when a ``%(name)s`` placeholder names a parameter
                not in ``QUERY_PARAMS`` (``""`` included, for ``%()s``); or when any other ``%`` sequence but ``%%`` is
                found, such as ``%s``, a lone ``%`` or ``%(source_id)d``, naming its position.

        """
        if not self.sql.strip():
            raise ValueError("an id query needs SQL")
        unknown: set[str] = set()
        for match in _PERCENT.finditer(self.sql):
            name = match.group("name")
            if match.group() == "%%":
                continue
            if name is None or match.group("format") != "s":
                raise ValueError(
                    f"an id query takes only %(name)s parameters and %% for a literal %, not {match.group()!r} at "
                    f"index {match.start()}"
                )
            if name not in QUERY_PARAMS:
                unknown.add(name)
        if unknown:
            raise ValueError(f"an id query takes only {sorted(QUERY_PARAMS)}, not {sorted(unknown)}")


#: A job's id source: the records of a list endpoint, or a warehouse query.
type IdSource = FromList | FromQuery
