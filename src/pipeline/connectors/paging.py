"""Request pages from a connector endpoint and find the cursor of the page after each response.

Public API:
    Paginator: The protocol every paging style implements: put a cursor on a request, read the next cursor back.
    NoPaging: The paginator for an endpoint that returns everything in one response.
    CursorPaginator: The paginator for an endpoint that returns an opaque next-page token.
    HasMore: An end rule that reads a boolean "more pages follow" flag from the body.
    ShortPage: An end rule that treats a page with fewer records than the page size as the last.
    TotalCount: An end rule that compares the pages fetched so far with a total record count in the body.
    EndRule: The union of the three end rules ``PageNumberPaginator`` accepts.
    PageNumberPaginator: The paginator for an endpoint paged by page number and page size.

"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from pipeline.connectors.spec import ConnectorRequest, EndpointSpec, KeyPath, Place, check_key_path, value_at


class Paginator(Protocol):
    """Describe how an endpoint's pages are requested and how the cursor of the next page is found.

    A cursor is a string, or ``None`` for the first page. The cursor ``next_cursor`` returns is the one the caller
    passes to ``apply`` for the following request, and ``None`` from ``next_cursor`` means there is no next page.

    """

    def apply(self, request: ConnectorRequest, cursor: str | None) -> ConnectorRequest:
        """Return the request for the page at a cursor.

        Preconditions:
            ``cursor`` is ``None`` or a value this paginator's ``next_cursor`` returned.

        Args:
            request: The request without paging parameters.
            cursor: The page to request, or ``None`` for the first page.

        Returns:
            A copy of ``request`` with the paging parameters added.

        Raises:
            ValueError: Defined by each implementation.

        """
        ...

    def next_cursor(self, body: Mapping[str, Any], cursor_in: str | None, *, spec: EndpointSpec) -> str | None:
        """Return the cursor of the page after a response, or ``None`` when it was the last page.

        Preconditions:
            ``body`` is the response to the request ``apply`` built from ``cursor_in``.

        Args:
            body: The decoded response body.
            cursor_in: The cursor the response was requested with.
            spec: The endpoint, used to read records and to build errors.

        Returns:
            The next page's cursor, or ``None``.

        Raises:
            ClientErrorUnhandled: Defined by each implementation, for a body that cannot be read.

        """
        ...


@dataclass(frozen=True, slots=True)
class NoPaging:
    """Page an endpoint that returns everything in one response.

    It holds no state and guarantees nothing beyond its methods: no parameters are added and there is no next page.

    """

    def apply(self, request: ConnectorRequest, cursor: str | None) -> ConnectorRequest:
        """Return the request unchanged, refusing any cursor.

        Preconditions:
            None.

        Args:
            request: The request to send.
            cursor: Must be ``None``.

        Returns:
            ``request`` itself.

        Raises:
            ValueError: When ``cursor`` is not ``None``.

        """
        if cursor is not None:
            raise ValueError(f"an unpaged endpoint has no cursor to resume from: {cursor!r}")
        return request

    def next_cursor(self, body: Mapping[str, Any], cursor_in: str | None, *, spec: EndpointSpec) -> str | None:
        """Return ``None``: an unpaged response is always the last page.

        Preconditions:
            None.

        Args:
            body: The response body, not read.
            cursor_in: The cursor sent, not read.
            spec: The endpoint, not read.

        Returns:
            None.

        Raises:
            Nothing.

        """
        return None


@dataclass(frozen=True, slots=True)
class CursorPaginator:
    """Page an endpoint that returns an opaque token for the next page.

    Every request carries the page size; every request after the first also carries the token. Construction
    guarantees that ``size`` is positive, that ``token_param`` and ``size_param`` are non-empty and different, and
    that ``next_at`` and any ``has_more_at`` are valid key paths.

    Attributes:
        token_param (str): The parameter name (one-key path) the token is sent under.
        size_param (str): The parameter name (one-key path) the page size is sent under.
        size (int): The page size requested, at least 1.
        next_at (KeyPath): Where the next token is in a response body.
        has_more_at (KeyPath | None): Where a boolean "more pages follow" flag is, or ``None`` to end on a missing,
            ``null`` or empty token instead.
        place (Place): Whether both parameters go in the query or the body. Defaults to the query.

    """

    token_param: str  # e.g. "page_token"
    size_param: str  # e.g. "page_size"
    size: int
    next_at: KeyPath  # e.g. ("data", "next_page_token")
    has_more_at: KeyPath | None = None  # set when the connector also says has_more; it then decides the end
    place: Place = Place.QUERY

    def __post_init__(self) -> None:
        """Reject a non-positive page size or an invalid key path.

        Preconditions:
            None.

        Returns:
            None.

        Raises:
            ValueError: When ``size`` is less than 1, when ``token_param`` or ``size_param`` is empty or they are
                equal, or from ``check_key_path`` when ``next_at`` or a set ``has_more_at`` is empty or has an empty
                key.

        """
        if self.size < 1:
            raise ValueError(f"page size must be positive: {self.size}")
        # a bad name would otherwise fail only when a task runs, after page 1 is saved
        if not self.token_param or not self.size_param or self.token_param == self.size_param:
            raise ValueError(f"parameter names must be non-empty and differ: {self.token_param!r}, {self.size_param!r}")
        check_key_path(self.next_at)
        if self.has_more_at is not None:
            check_key_path(self.has_more_at)

    def apply(self, request: ConnectorRequest, cursor: str | None) -> ConnectorRequest:
        """Return the request with the page size, and the token when there is one.

        Preconditions:
            None.

        Args:
            request: The request without paging parameters.
            cursor: The token of the page to request, or ``None`` for the first page.

        Returns:
            A copy of ``request`` with ``size_param`` set, and ``token_param`` set to ``cursor`` when it is not
            ``None``.

        Raises:
            ValueError: From ``ConnectorRequest.with_param`` when the request already has one of the names, or (in the
                body) the request is a GET.
            Exception: In the body, whatever ``copy.deepcopy`` raises through ``with_param``.

        """
        request = request.with_param(self.place, (self.size_param,), self.size)
        if cursor is None:  # first page: no token
            return request
        return request.with_param(self.place, (self.token_param,), cursor)

    def next_cursor(self, body: Mapping[str, Any], cursor_in: str | None, *, spec: EndpointSpec) -> str | None:
        """Return the next token from a response, or ``None`` when it was the last page.

        With ``has_more_at`` set, a false flag ends paging whatever the token, and a true flag requires a token.
        Without it, a missing, ``null`` or empty token ends paging. A token equal to ``cursor_in`` is rejected, but
        only that one-step repeat is detected: a connector cycling through two or more tokens is not. Records are not
        read.

        Preconditions:
            None.

        Args:
            body: The decoded response body.
            cursor_in: The token the response was requested with, or ``None`` for the first page.
            spec: The endpoint, used to build errors.

        Returns:
            The non-empty next token, or ``None``.

        Raises:
            ClientErrorUnhandled: From ``_flag`` when the flag is missing or not a bool; when the token is present
                and not ``None`` but not a string; when the flag is true and the token is missing, ``null`` or empty;
                or when the token equals ``cursor_in``.

        """
        if self.has_more_at is not None and not _flag(body, self.has_more_at, spec):
            return None
        try:
            token = value_at(body, self.next_at)
        except KeyError:
            token = None
        if token is not None and not isinstance(token, str):
            raise spec.unhandled(f"next cursor is a {type(token).__name__}, not a string")
        if not token:
            if self.has_more_at is not None:
                raise spec.unhandled("has_more is true but there is no next cursor")
            return None
        if token == cursor_in:
            raise spec.unhandled(f"connector returned the cursor just sent: {token!r}")
        return token


@dataclass(frozen=True, slots=True)
class HasMore:
    """End page-number paging when a boolean flag in the body is false.

    Construction guarantees that ``at`` is a valid key path.

    Attributes:
        at (KeyPath): Where the flag is in a response body.

    """

    at: KeyPath  # a boolean; false ends paging

    def __post_init__(self) -> None:
        """Reject an invalid key path.

        Preconditions:
            None.

        Returns:
            None.

        Raises:
            ValueError: From ``check_key_path`` when ``at`` is empty or has an empty key.

        """
        check_key_path(self.at)


@dataclass(frozen=True, slots=True)
class ShortPage:
    """End page-number paging at the first page with fewer records than the page size.

    A connector that caps its page size below the size requested returns only short pages, so paging ends after the
    first page without an error.

    """

    pass  # fewer records than the page size ends paging


@dataclass(frozen=True, slots=True)
class TotalCount:
    """End page-number paging once the pages requested so far cover a total record count in the body.

    The count is read from ``at``, independently of ``EndpointSpec.total_at``. Construction guarantees that ``at`` is
    a valid key path.

    Attributes:
        at (KeyPath): Where the total count is in a response body.

    """

    at: KeyPath  # the total row count; paging ends once the pages fetched cover it

    def __post_init__(self) -> None:
        """Reject an invalid key path.

        Preconditions:
            None.

        Returns:
            None.

        Raises:
            ValueError: From ``check_key_path`` when ``at`` is empty or has an empty key.

        """
        check_key_path(self.at)


#: How a ``PageNumberPaginator`` decides that a page was the last: a flag in the body, a short page, or a total
#: count in the body.
type EndRule = HasMore | ShortPage | TotalCount


@dataclass(frozen=True, slots=True)
class PageNumberPaginator:
    """Page an endpoint by page number and page size, with an end rule deciding when to stop.

    The cursor is the page number as a decimal string. Construction guarantees that ``size`` is positive,
    ``first_page`` is not negative, and ``page_param`` and ``size_param`` are non-empty and different.

    Attributes:
        page_param (str): The parameter name (one-key path) the page number is sent under.
        size_param (str): The parameter name (one-key path) the page size is sent under.
        size (int): The page size requested, at least 1.
        end (EndRule): How the last page is recognised.
        place (Place): Whether both parameters go in the query or the body. Defaults to the body.
        first_page (int): The number of the first page, at least 0. Defaults to 1.

    """

    page_param: str  # e.g. "page_number", or "page" on Cruva's ads endpoints
    size_param: str  # e.g. "page_size"
    size: int
    end: EndRule
    place: Place = Place.BODY
    first_page: int = 1

    def __post_init__(self) -> None:
        """Reject a non-positive page size or a negative first page.

        Preconditions:
            None.

        Returns:
            None.

        Raises:
            ValueError: When ``size`` is less than 1, ``first_page`` is less than 0, or ``page_param`` or
                ``size_param`` is empty or they are equal.

        """
        if self.size < 1:
            raise ValueError(f"page size must be positive: {self.size}")
        if self.first_page < 0:
            raise ValueError(f"first page must not be negative: {self.first_page}")
        if not self.page_param or not self.size_param or self.page_param == self.size_param:
            raise ValueError(f"parameter names must be non-empty and differ: {self.page_param!r}, {self.size_param!r}")

    def apply(self, request: ConnectorRequest, cursor: str | None) -> ConnectorRequest:
        """Return the request with the page number and page size.

        Preconditions:
            None.

        Args:
            request: The request without paging parameters.
            cursor: The page number as a decimal string, or ``None`` for ``first_page``.

        Returns:
            A copy of ``request`` with ``page_param`` and ``size_param`` set.

        Raises:
            ValueError: From ``_page`` when ``cursor`` is not a page number of at least ``first_page``; from
                ``ConnectorRequest.with_param`` when the request already has that name, or (in the body) the request is
                a GET.
            Exception: In the body, whatever ``copy.deepcopy`` raises through ``with_param``.

        """
        page = self._page(cursor)
        return request.with_param(self.place, (self.page_param,), page).with_param(
            self.place, (self.size_param,), self.size
        )

    def next_cursor(self, body: Mapping[str, Any], cursor_in: str | None, *, spec: EndpointSpec) -> str | None:
        """Return the next page number from a response, or ``None`` when the end rule says it was the last page.

        ``ShortPage`` continues while the page holds at least ``size`` records; ``HasMore`` while the flag is true;
        ``TotalCount`` while the number of pages from ``first_page`` to this one, times ``size``, is below the total.
        An empty page that the rule says is not the last is rejected. ``ShortPage`` counts records strictly, because
        the count is its only end signal; ``HasMore`` and ``TotalCount`` decide first and then count with
        ``last_page`` set when there is no next page, so a last page may leave out its records or send them as
        ``null``.

        Preconditions:
            None.

        Args:
            body: The decoded response body.
            cursor_in: The page number the response was requested with, or ``None`` for ``first_page``.
            spec: The endpoint, used to count records and to build errors.

        Returns:
            The next page number as a decimal string, or ``None``.

        Raises:
            ValueError: From ``_page`` when ``cursor_in`` is not a page number of at least ``first_page``.
            ClientErrorUnhandled: From ``EndpointSpec.record_count`` when the records are missing, ``null`` or not
                a list, except a missing or ``null`` list on a last page under ``HasMore`` or ``TotalCount``; from
                ``_flag`` or ``_total`` when the end rule's value is missing or invalid; or when the page is empty but
                the end rule says more pages follow.

        """
        page = self._page(cursor_in)
        match self.end:
            case ShortPage():
                count = spec.record_count(body)  # the count is the only end signal, so a missing list never ends
                more = count >= self.size
            case HasMore(at):
                more = _flag(body, at, spec)
                count = spec.record_count(body, last_page=not more)
            case TotalCount(at):
                more = (page - self.first_page + 1) * self.size < _total(body, at, spec)
                count = spec.record_count(body, last_page=not more)
        if not more:
            return None
        if count == 0:  # the connector says more pages follow but sent nothing: paging would never end
            raise spec.unhandled(f"page {page} is empty but the end rule {self.end} says more pages follow")
        return str(page + 1)

    def _page(self, cursor: str | None) -> int:
        """Return the page number a cursor names.

        Preconditions:
            None.

        Args:
            cursor: An ASCII decimal string, or ``None`` for ``first_page``. Leading zeros are accepted.

        Returns:
            ``first_page`` for ``None``, otherwise the cursor's value.

        Raises:
            ValueError: When ``cursor`` is not all ASCII digits (including empty), is below ``first_page``, or has
                more digits than ``int`` converts.

        """
        if cursor is None:
            return self.first_page
        if not (cursor.isascii() and cursor.isdigit()) or int(cursor) < self.first_page:
            raise ValueError(f"not a page number cursor: {cursor!r}")
        return int(cursor)


def _flag(body: Mapping[str, Any], path: KeyPath, spec: EndpointSpec) -> bool:
    """Return the boolean "more pages follow" flag at a key path.

    Preconditions:
        None.

    Args:
        body: The decoded response body.
        path: Where the flag is.
        spec: The endpoint, used to build errors.

    Returns:
        The flag.

    Raises:
        ClientErrorUnhandled: When nothing is at ``path`` or the value is not a bool.

    """
    try:
        value = value_at(body, path)
    except KeyError:
        raise spec.unhandled(f"no has_more flag at {'.'.join(path)}") from None
    if not isinstance(value, bool):
        raise spec.unhandled(f"has_more at {'.'.join(path)} is a {type(value).__name__}, not a bool")
    return value


def _total(body: Mapping[str, Any], path: KeyPath, spec: EndpointSpec) -> int:
    """Return the total record count at a key path.

    Preconditions:
        None.

    Args:
        body: The decoded response body.
        path: Where the count is.
        spec: The endpoint, used to build errors.

    Returns:
        The non-negative integer count.

    Raises:
        ClientErrorUnhandled: When nothing is at ``path`` or the value is not a non-negative ``int`` (booleans, floats,
            strings and ``None`` are rejected).

    """
    try:
        value = value_at(body, path)
    except KeyError:
        raise spec.unhandled(f"no total count at {'.'.join(path)}") from None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise spec.unhandled(f"total count at {'.'.join(path)} is not a count: {value!r}")
    return value
