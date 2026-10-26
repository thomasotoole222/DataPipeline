"""Describe connector API endpoints and build the HTTP requests made to them.

Public API:
    HttpMethod: The HTTP methods a connector request may use, ``"GET"`` or ``"POST"``.
    KeyPath: A tuple of keys naming a location inside nested JSON objects.
    Place: Where a request parameter goes, the query string or the JSON body.
    value_at: Return the value at a key path in a nested mapping.
    check_key_path: Reject an empty key path or one containing an empty key.
    ConnectorRequest: A validated method, path, query and body for one HTTP call to a connector.
    EndpointSpec: A validated description of one connector endpoint and how to read its responses.
    EndpointRegistry: A read-only lookup of endpoint specs by name.

"""

import copy
import string
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal, get_args
from urllib.parse import quote

from pipeline.errors import ClientErrorUnhandled

if TYPE_CHECKING:
    from pipeline.connectors.paging import Paginator
    from pipeline.connectors.windows import Window

#: The HTTP methods a connector request may use. Only GET and POST are listed because no connector endpoint
#: needs others; ``_METHODS`` is derived from this alias so the two cannot drift.
type HttpMethod = Literal["GET", "POST"]
#: A path of keys into nested JSON objects, outermost first. Lists along the way are not indexed.
type KeyPath = tuple[str, ...]

#: The runtime set of ``HttpMethod`` values, taken from the alias with ``get_args`` so the check in
#: ``ConnectorRequest.__post_init__`` always matches the type.
_METHODS: Final = frozenset(get_args(HttpMethod.__value__))
#: An empty, read-only mapping used as the default of ``EndpointSpec.request``, so the default cannot be
#: mutated and shared between calls.
_NO_PATH_PARAMS: Final[Mapping[str, str]] = MappingProxyType({})


class Place(StrEnum):
    """Name where a request parameter is placed.

    Members are strings, so they compare equal to ``"query"`` and ``"body"``.

    Attributes:
        QUERY: The parameter goes in the URL query string, as a single key with a string value.
        BODY: The parameter goes in the JSON body, at a key path.

    """

    QUERY = "query"
    BODY = "body"


def value_at(body: Mapping[str, Any], path: KeyPath) -> Any:
    """Return the value at a key path in a nested mapping.

    An empty path returns ``body`` itself. Only mappings are traversed: a list or scalar part-way along the path,
    including at the top when outside JSON is not an object, is treated as a missing key.

    Preconditions:
        None.

    Args:
        body: The outermost mapping, usually a decoded JSON response.
        path: The keys to follow, outermost first.

    Returns:
        The value found, which may be any JSON value including ``None``.

    Raises:
        KeyError: With ``path`` as its argument, when a step's container is not a mapping or lacks the key.

    """
    value: Any = body
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise KeyError(path)
        value = value[key]
    return value


def check_key_path(path: KeyPath) -> None:
    """Reject a key path that is empty or contains an empty key.

    Preconditions:
        None.

    Args:
        path: The key path to check.

    Returns:
        None.

    Raises:
        ValueError: When ``path`` is empty or any key in it is ``""``.

    """
    if not path or not all(path):
        raise ValueError(f"a key path needs at least one key and no empty keys: {path!r}")


@dataclass(frozen=True, slots=True)
class ConnectorRequest:
    """Hold the method, path, query and body of one HTTP call to a connector.

    Construction guarantees that the method is GET or POST, the path starts with ``/``, and a GET request has no
    body. ``query`` is replaced by a shallow ``dict`` copy and ``body`` by a deep ``dict`` copy, so later changes to
    the caller's mappings do not reach the request. Callers must not change ``query`` or ``body`` in place and must
    extend a request only through the ``with_`` builders; nothing hashes a request, and ``hash()`` raises
    ``TypeError``. Query keys are not checked for emptiness, and a POST may have no body.

    Attributes:
        method (HttpMethod): The HTTP method.
        path (str): The concrete request path, starting with ``/``, with path parameters already substituted.
        query (Mapping[str, str]): Query string parameters, held as a ``dict``.
        body (Mapping[str, Any] | None): The JSON body, held as a deep-copied ``dict``, or ``None``.
        source_scoped (bool): Whether the request is made on behalf of a source. Defaults to ``True``.

    """

    method: HttpMethod
    path: str  # filled, no template fields left
    query: Mapping[str, str]
    body: Mapping[str, Any] | None  # None sends no body
    source_scoped: bool = True  # False: the client sends no source identifier

    def __post_init__(self) -> None:
        """Validate the fields and replace ``query`` and ``body`` with private copies.

        Preconditions:
            None.

        Returns:
            None.

        Raises:
            ValueError: When ``method`` is not GET or POST, ``path`` does not start with ``/``, or a GET has a body.
            Exception: Whatever ``copy.deepcopy`` raises for a body value that cannot be deep-copied.

        """
        if self.method not in _METHODS:
            raise ValueError(f"unsupported method: {self.method!r}")
        if not self.path.startswith("/"):
            raise ValueError(f"path must start with '/': {self.path!r}")
        if self.method == "GET" and self.body is not None:
            raise ValueError("a GET request carries no body")
        # own copies, so nothing the caller still holds can change a built request
        object.__setattr__(self, "query", dict(self.query))
        if self.body is not None:
            object.__setattr__(self, "body", copy.deepcopy(dict(self.body)))

    def with_query(self, params: Mapping[str, str]) -> ConnectorRequest:
        """Return a copy of this request with query parameters added.

        Preconditions:
            None.

        Args:
            params: The parameters to add. Empty names are accepted.

        Returns:
            A new ``ConnectorRequest`` with the merged query and the same method, path, body and scope.

        Raises:
            ValueError: When any key in ``params`` is already in the query.

        """
        clash = params.keys() & self.query.keys()
        if clash:
            raise ValueError(f"query parameters already set: {sorted(clash)}")
        return ConnectorRequest(self.method, self.path, {**self.query, **params}, self.body, self.source_scoped)

    def with_body_at(self, path: KeyPath, value: Any) -> ConnectorRequest:
        """Return a copy of this request with a value set at a new key path in the body.

        A POST with no body starts from an empty object, and missing intermediate objects are created. The new request's
        construction deep-copies the value, so the caller's object is not shared.

        Preconditions:
            None.

        Args:
            path: Where to put the value, outermost key first.
            value: The JSON value to set.

        Returns:
            A new ``ConnectorRequest`` with the updated body and the same method, path, query and scope.

        Raises:
            ValueError: From ``check_key_path`` when ``path`` is empty or has an empty key (checked before the method);
                when the request is a GET; or from ``_set_new`` when the final key is already set or an intermediate
                key holds something that is not a mapping.
            Exception: Whatever ``copy.deepcopy`` raises, from ``__post_init__``, for a value that cannot be
                deep-copied.

        """
        check_key_path(path)
        if self.method == "GET":
            raise ValueError("a GET request carries no body")
        body = _set_new(self.body or {}, path, value)
        return ConnectorRequest(self.method, self.path, self.query, body, self.source_scoped)

    def with_param(self, place: Place, path: KeyPath, value: str | int) -> ConnectorRequest:
        """Return a copy of this request with one parameter added at the given place.

        In the body the value keeps its type, so an ``int`` stays a JSON number. In the query it is converted with
        ``str``.

        Preconditions:
            None.

        Args:
            place: Whether the parameter goes in the query or the body.
            path: The body key path, or a one-key path naming the query parameter.
            value: The parameter value.

        Returns:
            A new ``ConnectorRequest`` with the parameter added.

        Raises:
            ValueError: From ``check_key_path`` when ``path`` is empty or has an empty key, in either place; then, for
                ``Place.QUERY``, when ``path`` has more than one key, or from ``with_query`` when the name is already
                set; for ``Place.BODY``, anything else ``with_body_at`` raises.
            Exception: For ``Place.BODY``, whatever ``copy.deepcopy`` raises through ``with_body_at``.

        """
        check_key_path(path)  # both places reject the same names, an empty one included
        if place is Place.BODY:
            return self.with_body_at(path, value)
        if len(path) != 1:
            raise ValueError(f"a query parameter is a single key, not {path!r}")
        return self.with_query({path[0]: str(value)})


def _set_new(body: Mapping[str, Any], path: KeyPath, value: Any) -> dict[str, Any]:
    """Return a copy of a body with a value set at a key path that must not already hold a value.

    Only the mappings along ``path`` are copied, each into a new ``dict``. ``body`` is not mutated, and branches off
    the path are shared with it. Missing intermediate keys become new objects.

    Preconditions:
        ``path`` is non-empty; ``with_body_at`` ensures this with ``check_key_path``.

    Args:
        body: The body to copy.
        path: Where to set the value, outermost key first.
        value: The value to set.

    Returns:
        A new ``dict`` with the value set.

    Raises:
        ValueError: When the final key is already present (even holding ``None``), or an intermediate key holds
            something that is not a mapping.
        IndexError: When ``path`` is empty, which breaks the precondition.

    """
    key, rest = path[0], path[1:]
    out = dict(body)
    if not rest:
        if key in out:
            raise ValueError(f"body key already set: {key!r}")
        out[key] = value
        return out
    inner = out.get(key, {})
    if not isinstance(inner, Mapping):
        raise ValueError(f"body key {key!r} holds a {type(inner).__name__}, not an object")
    out[key] = _set_new(inner, rest, value)
    return out


def _template_fields(template: str) -> frozenset[str]:
    """Return the field names in a path template, rejecting anything but plain named fields.

    Escaped braces (``{{`` and ``}}``) are literal text, not fields.

    Preconditions:
        None.

    Args:
        template: A ``str.format`` style path such as ``/products/{product_id}``.

    Returns:
        The distinct field names, empty when the template has none.

    Raises:
        ValueError: From ``string.Formatter.parse`` for unbalanced braces; or when a field is positional, is not an
            identifier (such as ``{a.b}`` or ``{a[0]}``), or has a conversion or a non-empty format spec.

    """
    names: set[str] = set()
    for _, name, spec, conversion in string.Formatter().parse(template):
        if name is None:
            continue
        if not name.isidentifier() or spec or conversion:
            raise ValueError(f"path fields must be plain names like {{product_id}}: {template!r}")
        names.add(name)
    return frozenset(names)


@dataclass(frozen=True, slots=True)
class EndpointSpec:
    """Describe one connector endpoint: how to build its requests and where to read its responses.

    Construction guarantees: ``name`` is ``<connector>.<endpoint>`` with a non-empty connector and endpoint; ``path``
    is a template with only plain named fields and starts with ``/``; the method is GET or POST and a GET has no
    ``fixed_body``; ``records_at`` and any ``total_at`` are valid key paths; ``max_rows`` is positive when set; and
    ``windows`` is frozen into a read-only mapping. Callers must not change ``fixed_query`` or ``fixed_body`` after
    construction, because every later ``request()`` would pick up the change; nothing hashes a spec, and ``hash()``
    raises ``TypeError``. Not checked: window names, and whether the paginator and windows use places and parameter
    names that suit the method and each other.

    Attributes:
        name (str): The registry name, ``<connector>.<endpoint>``, for example ``"tiktok_shop.orders_search"``.
        connector (str): The connector the endpoint belongs to, for example ``"tiktok_shop"``.
        method (HttpMethod): The HTTP method.
        path (str): The path template, starting with ``/``.
        paginator (Paginator): How pages are requested and the next cursor found.
        records_at (KeyPath): Where the list of records is in a response body.
        fixed_query (Mapping[str, str]): Query parameters sent on every request. Defaults to empty.
        fixed_body (Mapping[str, Any] | None): The body every request starts from, or ``None``.
        windows (Mapping[str, Window]): Named ways of applying a time frame, read-only after construction.
        max_rows (int | None): A positive row limit, or ``None``. Not used on this page.
        total_at (KeyPath | None): Where a total record count is in a response body, or ``None``.
        source_scoped (bool): Whether requests are made on behalf of a source, so the client sends the source's
            identifier. ``False`` for an account-wide endpoint. Defaults to ``True``.

    """

    name: str  # "<connector>.<endpoint>", e.g. "tiktok_shop.orders_search"
    connector: str
    method: HttpMethod
    path: str  # template, e.g. "/product/202309/products/{product_id}"
    paginator: Paginator
    records_at: KeyPath  # where the record list sits in the body
    fixed_query: Mapping[str, str] = field(default_factory=dict)
    fixed_body: Mapping[str, Any] | None = None
    windows: Mapping[str, Window] = field(default_factory=dict)  # by time field, e.g. "update", "create"
    max_rows: int | None = None  # the connector's row cap for one frame; None: no cap
    total_at: KeyPath | None = None  # where the connector reports the whole frame's row total; None: not reported
    source_scoped: bool = True  # False: an account-wide endpoint, sent without a source identifier

    @property
    def endpoint(self) -> str:
        """The endpoint's name within its connector: ``name`` without the ``<connector>.`` prefix.

        Preconditions:
            None; construction guarantees ``name`` starts with ``<connector>.``.

        Returns:
            For example ``"orders_search"`` for ``"tiktok_shop.orders_search"``. Never empty.

        Raises:
            Nothing.

        """
        return self.name.removeprefix(f"{self.connector}.")

    def __post_init__(self) -> None:
        """Validate the spec and freeze ``windows``.

        The method, path prefix and GET-body rules are checked by building a throwaway ``ConnectorRequest`` from the
        template.

        Preconditions:
            None.

        Returns:
            None.

        Raises:
            ValueError: When the name is not ``<connector>.<endpoint>``; from ``_template_fields`` for a bad path
                template; from ``check_key_path`` for a bad ``records_at`` or ``total_at``; when ``max_rows`` is less
                than 1; or from ``ConnectorRequest`` when the method is unsupported, the path does not start with
                ``/``, or a GET has a body.
            Exception: Whatever ``copy.deepcopy`` raises, from ``ConnectorRequest``, for a ``fixed_body`` value that
                cannot be deep-copied.

        """
        if not self.connector or not self.name.startswith(f"{self.connector}.") or self.name == f"{self.connector}.":
            raise ValueError(f"spec name must be '<connector>.<endpoint>': {self.name!r} for {self.connector!r}")
        _template_fields(self.path)
        check_key_path(self.records_at)
        if self.max_rows is not None and self.max_rows < 1:
            raise ValueError(f"max_rows must be positive: {self.max_rows}")
        if self.total_at is not None:
            check_key_path(self.total_at)
        object.__setattr__(self, "windows", MappingProxyType(dict(self.windows)))
        # builds the bare request once, so a bad method, path or GET body fails here, not on first use
        ConnectorRequest(self.method, self.path, self.fixed_query, self.fixed_body, self.source_scoped)

    def request(self, path_params: Mapping[str, str] = _NO_PATH_PARAMS) -> ConnectorRequest:
        """Return the first request for this endpoint, with path parameters substituted.

        Each value is percent-encoded with no safe characters, so ``/`` becomes ``%2F``. The template was validated by
        ``__post_init__``, so formatting cannot fail once the keys match.

        Preconditions:
            None.

        Args:
            path_params: One value for each field in the path template, and no others. Defaults to none.

        Returns:
            A new ``ConnectorRequest`` carrying the fixed query, fixed body and scope.

        Raises:
            ValueError: When the keys of ``path_params`` differ from the template's fields, or when any value in
                ``path_params`` is empty.

        """
        expected = _template_fields(self.path)
        if path_params.keys() != expected:
            raise ValueError(f"{self.name}: path needs {sorted(expected)}, got {sorted(path_params)}")
        # an empty value would turn an item path such as /products/{product_id} into the collection path
        empty = sorted(name for name, value in path_params.items() if not value)
        if empty:
            raise ValueError(f"{self.name}: empty path values for {empty}")
        path = self.path.format_map({k: quote(v, safe="") for k, v in path_params.items()})
        return ConnectorRequest(self.method, path, self.fixed_query, self.fixed_body, self.source_scoped)

    def records(self, body: Mapping[str, Any]) -> list[Any]:
        """Return the records list from a response body.

        The list returned is the body's own object, not a copy.

        Preconditions:
            None.

        Args:
            body: A decoded response body.

        Returns:
            The list at ``records_at``.

        Raises:
            ClientErrorUnhandled: From ``unhandled`` when nothing is at ``records_at`` or the value there is not a list,
                including ``None``.

        """
        try:
            found = value_at(body, self.records_at)
        except KeyError:
            raise self.unhandled(f"no records at {'.'.join(self.records_at)}") from None
        if not isinstance(found, list):
            raise self.unhandled(f"records at {'.'.join(self.records_at)} are a {type(found).__name__}, not a list")
        return found

    def record_count(self, body: Mapping[str, Any], *, last_page: bool = False) -> int:
        """Return the number of records in a response body.

        On the last page, nothing at ``records_at`` (missing as ``value_at`` defines it) or a ``None`` value there
        counts as 0. Any other value that is not a list still raises, even on the last page.

        Preconditions:
            None.

        Args:
            body: A decoded response body.
            last_page: Whether this body is the final page.

        Returns:
            The length of the records list, or 0 on a last page with nothing or ``null`` at ``records_at``.

        Raises:
            ClientErrorUnhandled: From ``records`` when the records are missing or ``null`` on a page that is not the
                last, or the value is something other than a list.

        """
        if last_page:
            try:
                found = value_at(body, self.records_at)
            except KeyError:
                found = None
            if found is None:
                return 0  # a connector may leave out, or send as null, an empty record list on its last page
        return len(self.records(body))

    def total(self, body: Mapping[str, Any]) -> int | None:
        """Return the total record count reported in a response body, if the endpoint reports one.

        Preconditions:
            None.

        Args:
            body: A decoded response body.

        Returns:
            The non-negative integer at ``total_at``, or ``None`` when ``total_at`` is not set.

        Raises:
            ClientErrorUnhandled: From ``unhandled`` when nothing is at ``total_at``, or the value is not a non-negative
                ``int`` (booleans, floats, strings and ``None`` are rejected).

        """
        if self.total_at is None:
            return None
        where = ".".join(self.total_at)
        try:
            found = value_at(body, self.total_at)
        except KeyError:
            raise self.unhandled(f"no total at {where}") from None
        # Trust boundary: a count from the network
        if not isinstance(found, int) or isinstance(found, bool) or found < 0:
            raise self.unhandled(f"total at {where} is not a count: {found!r}")
        return found

    def unhandled(self, message: str) -> ClientErrorUnhandled:
        """Build an error for a successful response this spec cannot read, without raising it.

        The error carries the path template, not the concrete path, with status 200 and no code or request id.

        Preconditions:
            None.

        Args:
            message: What was wrong with the response.

        Returns:
            A ``ClientErrorUnhandled`` for this spec's connector.

        Raises:
            Nothing.

        """
        # a 200 whose body breaks this spec's expectations
        return ClientErrorUnhandled(
            connector=self.connector, path=self.path, status=200, code=None, message=message, request_id=None
        )


@dataclass(frozen=True, slots=True)
class EndpointRegistry:
    """Look up endpoint specs by name.

    Construction guarantees that every key equals its spec's ``name``, and freezes ``specs`` into a read-only
    mapping.

    Attributes:
        specs (Mapping[str, EndpointSpec]): The specs keyed by name, read-only.

    """

    specs: Mapping[str, EndpointSpec]

    def __post_init__(self) -> None:
        """Check that every key names its spec, and freeze the mapping.

        Preconditions:
            None.

        Returns:
            None.

        Raises:
            ValueError: When a key differs from its spec's ``name``.

        """
        for name, spec in self.specs.items():
            if name != spec.name:
                raise ValueError(f"registry key {name!r} names spec {spec.name!r}")
        object.__setattr__(self, "specs", MappingProxyType(dict(self.specs)))

    @classmethod
    def of(cls, specs: Iterable[EndpointSpec]) -> EndpointRegistry:
        """Build a registry from specs, keyed by their names.

        Preconditions:
            None.

        Args:
            specs: The specs to register.

        Returns:
            A new ``EndpointRegistry``.

        Raises:
            ValueError: When two specs share a name.

        """
        by_name: dict[str, EndpointSpec] = {}
        for spec in specs:
            if spec.name in by_name:
                raise ValueError(f"two endpoint specs named {spec.name!r}")
            by_name[spec.name] = spec
        return cls(by_name)

    def get(self, name: str) -> EndpointSpec:
        """Return the spec with the given name.

        Preconditions:
            None.

        Args:
            name: The spec's ``<connector>.<endpoint>`` name.

        Returns:
            The matching ``EndpointSpec``.

        Raises:
            KeyError: With a message naming ``name``, when no spec has that name.

        """
        try:
            return self.specs[name]
        except KeyError:
            raise KeyError(f"no endpoint spec named {name!r}") from None
