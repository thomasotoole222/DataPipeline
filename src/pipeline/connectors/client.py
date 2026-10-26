"""The rules that classify connector error responses, and the HTTP client base that sends, records and retries requests.

Public API:
    CredentialProvider: Protocol for getting a source's credential for a connector, and reporting one it rejected.
    ConnectorClient: Protocol for a client that sends a connector request for a source and returns the response body.
    ErrorOutcome: The kinds of pipeline error a matched ``ErrorRule`` turns a connector error into.
    ErrorRule: A rule that matches a connector error by code, status and message keyword and names its outcome.
    map_error: Convert a connector error into the pipeline error named by the first rule that matches it.
    Fault: An error a connector reported inside a response: code, message and request id.
    HttpConnectorClient: Base class for HTTP connector clients that classify, record and retry each request.

"""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from functools import partial
from typing import Any, Protocol, assert_never
from uuid import UUID

import httpx

from pipeline.connectors.spec import ConnectorRequest
from pipeline.errors import (
    AuthBlocked,
    ClientErrorUnhandled,
    ConnectorResponseError,
    PermanentError,
    PipelineError,
    RateLimited,
    TokenRejected,
    TransientError,
    retries,
    retry_after_from,
    translates_errors,
)
from pipeline.telemetry import Instruments
from pipeline.timeutil import utc_now


class CredentialProvider[C](Protocol):
    """Provide a source's credential for a connector, and take back a credential the connector rejected.

    ``C`` is the credential type a client sends. The protocol states no error contract; ``HttpConnectorClient`` calls
    ``get`` once per attempt and ``reject`` after a ``TokenRejected``.

    """

    async def get(self, source_id: UUID, connector: str) -> C:
        """Return the credential to send with a request for a source to a connector.

        Preconditions:
            None stated by the protocol.

        Args:
            source_id: The source the request is made for.
            connector: Name of the connector, ``HttpConnectorClient.connector`` when called by that class.

        Returns:
            The credential.

        Raises:
            Not specified by the protocol. ``HttpConnectorClient._attempt`` calls it outside its error handling, so an
            exception is not recorded; a ``TransientError`` is retried by ``HttpConnectorClient.send``.

        """
        ...

    async def reject(self, source_id: UUID, connector: str, credential: C) -> None:
        """Report that a connector rejected a credential.

        ``HttpConnectorClient._attempt`` calls it when a request raises ``TokenRejected``, before re-raising that error.

        Preconditions:
            None stated by the protocol.

        Args:
            source_id: The source the rejected request was made for.
            connector: Name of the connector that rejected the credential.
            credential: The credential that ``get`` returned for that request.

        Returns:
            None.

        Raises:
            Not specified by the protocol. ``HttpConnectorClient._attempt`` lets it propagate, replacing the
            ``TokenRejected``; a ``TransientError`` is retried, so ``reject`` can be called again for the same
            credential.

        """
        ...


class ConnectorClient[C](Protocol):
    """Send connector requests for a source and return the response bodies.

    Attributes:
        connector (str): Name of the connector the client talks to.

    """

    connector: str

    async def send(self, request: ConnectorRequest, source_id: UUID) -> dict[str, Any]:
        """Send a request for a source and return the response body.

        Preconditions:
            None stated by the protocol.

        Args:
            request: The request to send.
            source_id: The source the request is made for.

        Returns:
            The response body as a JSON object.

        Raises:
            Not specified by the protocol; ``HttpConnectorClient.send`` documents its own.

        """
        ...


class ErrorOutcome(StrEnum):
    """Enumerate the pipeline errors that ``map_error`` builds for a matched rule.

    Each member is a string equal to its lowercase name. ``_result`` also uses the members as telemetry labels.

    Attributes:
        RATE_LIMITED: ``RateLimited``, with ``retry_after`` read from the ``Retry-After`` header.
        TRANSIENT: ``TransientError``.
        TOKEN_REJECTED: ``TokenRejected``; the client reports the credential to its ``CredentialProvider``.
        BLOCKED: ``AuthBlocked``.
        PERMANENT: ``PermanentError``.

    """

    RATE_LIMITED = "rate_limited"
    TRANSIENT = "transient"
    TOKEN_REJECTED = "token_rejected"  # noqa: S105 - an outcome name, not a secret
    BLOCKED = "blocked"
    PERMANENT = "permanent"


@dataclass(frozen=True, slots=True, kw_only=True)
class ErrorRule:
    """Match a connector error by code, status and message keyword, and name the outcome it maps to.

    A rule matches an error when every condition it sets holds: ``code`` equals the error's code; ``codeless`` requires
    the error to have no code; ``status`` equals the error's status, or contains it when ``status`` is a ``range``;
    ``keyword`` occurs in the error's message, compared after ``str.casefold``. A field left at ``None`` or ``False``
    matches any error.

    ``__post_init__`` enforces that at least one of ``code``, ``status`` and ``keyword`` is set (``codeless`` alone
    is not enough), that ``keyword`` is not empty, that ``codeless`` and ``code`` are not both set, and that a
    ``range`` for ``status`` is not empty.

    Attributes:
        outcome (ErrorOutcome): The kind of pipeline error ``map_error`` builds when this rule matches.
        code (int | None): Connector-specific error code the error must have, or ``None`` to match any.
        status (int | range | None): Status code, or range of status codes, the error must have, or ``None``.
        keyword (str | None): Text the error message must contain, ignoring case, or ``None``.
        codeless (bool): ``True`` if the error must have no code.

    """

    outcome: ErrorOutcome
    code: int | None = None
    status: int | range | None = None
    keyword: str | None = None
    # Matches only responses that carried no connector error code (HTTP-status fallbacks).
    codeless: bool = False

    def __post_init__(self) -> None:
        """Reject a rule that sets no condition or sets contradictory ones.

        Preconditions:
            None.

        Returns:
            None.

        Raises:
            ValueError: If ``code``, ``status`` and ``keyword`` are all ``None``; if ``codeless`` is ``True`` and
                ``code`` is set; if ``keyword`` is ``""``; or if ``status`` is an empty ``range``.

        """
        if self.code is None and self.status is None and self.keyword is None:
            raise ValueError("an error rule must set at least one of code, status or keyword")
        if self.codeless and self.code is not None:
            raise ValueError("a codeless rule cannot also match a code")
        if self.keyword == "":
            raise ValueError("keyword must not be empty")
        if isinstance(self.status, range) and not self.status:
            raise ValueError(f"status range is empty, so the rule could never match: {self.status!r}")

    def matches(self, err: ConnectorResponseError) -> bool:
        """Return whether a connector error meets every condition this rule sets.

        Preconditions:
            None.

        Args:
            err: The connector error to test.

        Returns:
            ``True`` if the error's code, status and message meet the conditions described on the class, else
            ``False``.

        Raises:
            None.

        """
        if self.code is not None and err.code != self.code:
            return False
        if self.codeless and err.code is not None:
            return False
        if isinstance(self.status, range):
            if err.status not in self.status:
                return False
        elif self.status is not None and err.status != self.status:
            return False
        return self.keyword is None or self.keyword.casefold() in err.message.casefold()


def map_error(rules: Sequence[ErrorRule], err: ConnectorResponseError, *, now: datetime | None = None) -> PipelineError:
    """Convert a connector error into the pipeline error named by the first rule that matches it.

    Rules are tried in order. With no match, the result is a ``ClientErrorUnhandled`` carrying the error's connector,
    path, status, code, message and request id; its headers are dropped. With a match, the result depends on the
    rule's outcome: ``RateLimited(str(err))`` with ``retry_after`` read from the error's ``Retry-After`` header and
    measured from ``now`` (``utc_now()`` when ``now`` is ``None``), ``TransientError(str(err))``,
    ``TokenRejected`` or ``AuthBlocked`` with the error's connector and its message as the reason, or
    ``PermanentError(str(err))``. The returned error's ``__cause__`` is not set.

    Preconditions:
        ``now``, if given, is timezone-aware.

    Args:
        rules: The rules to try, in order.
        err: The connector error to convert.
        now: Keyword-only. The moment a ``Retry-After`` HTTP date is measured from, or ``None`` for ``utc_now()``.

    Returns:
        The pipeline error described above. It is returned, not raised.

    Raises:
        TypeError: From ``retry_after_from``, if a ``RATE_LIMITED`` rule matches, the ``Retry-After`` value is an HTTP
            date and ``now`` is naive. ``HttpConnectorClient`` never passes ``now``, and ``utc_now()`` is aware.

    """
    rule = next((r for r in rules if r.matches(err)), None)
    if rule is None:
        return ClientErrorUnhandled(
            connector=err.connector,
            path=err.path,
            status=err.status,
            code=err.code,
            message=err.message,
            request_id=err.request_id,
        )
    match rule.outcome:
        case ErrorOutcome.RATE_LIMITED:
            return RateLimited(str(err), retry_after=retry_after_from(err.headers, now=now or utc_now()))
        case ErrorOutcome.TRANSIENT:
            return TransientError(str(err))
        case ErrorOutcome.TOKEN_REJECTED:
            return TokenRejected(connector=err.connector, reason=err.message)
        case ErrorOutcome.BLOCKED:
            return AuthBlocked(connector=err.connector, reason=err.message)
        case ErrorOutcome.PERMANENT:
            return PermanentError(str(err))
        case _:
            assert_never(rule.outcome)


@dataclass(frozen=True, slots=True)
class Fault:
    """Hold an error that a connector reported in a response, as returned by ``HttpConnectorClient.fault``.

    No invariants are enforced.

    Attributes:
        code (int | None): Connector-specific error code, or ``None`` if the response had none.
        message (str): The connector's error message.
        request_id (str | None): Request identifier reported by the connector, or ``None``.

    """

    code: int | None
    message: str
    request_id: str | None


class HttpConnectorClient[C](ABC):
    """Send connector requests over HTTP, classify errors by rule, record every attempt and retry transient failures.

    A subclass sets the class attributes ``connector`` and ``rules`` and implements ``build`` and ``fault``. Each
    attempt gets a credential from ``tokens`` and sends the request built by ``build``. Every attempt that reaches the
    exchange is recorded in ``instruments.connector_requests``; an attempt whose ``tokens.get`` raises, or whose
    ``build`` or ``fault`` raises something other than a ``PipelineError`` or ``httpx.HTTPError``, is not. Error
    responses are converted with ``map_error`` over ``rules``; a
    ``TokenRejected`` credential is reported to ``tokens``; ``TransientError`` is retried with the defaults of
    ``retries``.

    Attributes:
        connector (str): Class attribute set by the subclass. Name of the connector, used in errors and telemetry.
        rules (tuple[ErrorRule, ...]): Class attribute set by the subclass. Rules ``map_error`` tries in order.
        http (httpx.AsyncClient): The client requests are sent with.
        tokens (CredentialProvider[C]): Where credentials come from and rejected ones go.
        instruments (Instruments): Telemetry instruments; ``connector_requests`` counts attempts.

    """

    connector: str
    rules: tuple[ErrorRule, ...]

    def __init__(
        self,
        http: httpx.AsyncClient,
        tokens: CredentialProvider[C],
        instruments: Instruments,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """Store the dependencies and build the error-translating exchange and the retrying send.

        ``self.rules`` is read once here; the exchange keeps that tuple even if the attribute changes later.

        Preconditions:
            The subclass defines ``connector`` and ``rules``.

        Args:
            http: The client requests are sent with.
            tokens: Where credentials come from and rejected ones go.
            instruments: Telemetry instruments.
            sleep: Keyword-only. Awaited with each retry wait in seconds.

        Returns:
            None.

        Raises:
            AttributeError: If the subclass does not define ``connector`` or ``rules``.

        """
        self.http = http
        self.tokens = tokens
        self.instruments = instruments
        self._exchange = translates_errors(self.connector, partial(map_error, self.rules))(self._call)
        self._send = retries(sleep=sleep)(self._attempt)

    async def send(self, request: ConnectorRequest, source_id: UUID) -> dict[str, Any]:
        """Send a request for a source, retrying transient failures, and return the response body.

        Runs ``_attempt`` under ``retries`` with its defaults: up to 5 retries after the first attempt while it raises
        ``TransientError``, waiting with exponential backoff (1 s base, 60 s cap, up to 0.5 s jitter) through
        ``sleep``. Every attempt that reaches the exchange is recorded; an attempt whose ``tokens.get`` raises, or
        whose ``build`` or ``fault`` raises something other than a ``PipelineError`` or ``httpx.HTTPError``, is not.

        Preconditions:
            None.

        Args:
            request: The request to send.
            source_id: The source the request is made for.

        Returns:
            The JSON-object body of the first attempt whose response was 2xx and for which ``fault`` returned ``None``.

        Raises:
            TransientError: The last attempt's error once retries are used up. It covers ``RateLimited``, rule
                outcomes of ``TRANSIENT``, and ``httpx.TimeoutException``, ``httpx.NetworkError`` and
                ``httpx.RemoteProtocolError``, which are translated.
            RateLimited: At once, without retrying, when its ``retry_after`` exceeds 60 seconds.
            TokenRejected: After the credential is reported to ``tokens``. Not retried.
            AuthBlocked: From a ``BLOCKED`` rule. Not retried.
            PermanentError: From a ``PERMANENT`` rule. Not retried.
            ClientErrorUnhandled: When no rule matches an error, or a 2xx body is not a JSON object. Not retried.
            httpx.HTTPError: Any other ``httpx`` error, untranslated. Not retried.
            Any exception from ``tokens.get``, ``tokens.reject``, ``build``, ``fault`` or ``sleep`` propagates; a
                ``TransientError`` among them is retried.

        """
        return await self._send(request, source_id)

    @abstractmethod
    def build(self, request: ConnectorRequest, credential: C) -> httpx.Request:
        """Build the HTTP request for a connector request, authenticated with a credential.

        Preconditions:
            None on this page.

        Args:
            request: The connector request to send.
            credential: The credential ``tokens.get`` returned for this attempt.

        Returns:
            The request passed to ``http.send``.

        Raises:
            Whatever the implementation raises. Unless it is a ``PipelineError`` or ``httpx.HTTPError``, it escapes
            ``send`` without the attempt being recorded.

        """
        ...

    @abstractmethod
    def fault(self, response: httpx.Response, body: Mapping[str, Any]) -> Fault | None:
        """Return the error a response reports, or ``None`` if it reports none.

        ``_call`` calls it for every response whose body is a JSON object. For a 2xx response, ``None`` means success.
        For any other response, ``None`` makes ``_call`` use the first 500 characters of the response text as the
        message, with no code and no request id.

        Preconditions:
            None on this page.

        Args:
            response: The response received.
            body: The response body parsed as a JSON object.

        Returns:
            The reported error, or ``None``.

        Raises:
            Whatever the implementation raises. Unless it is a ``PipelineError`` or ``httpx.HTTPError``, it escapes
            ``send`` without the attempt being recorded.

        """
        ...

    async def _attempt(self, request: ConnectorRequest, source_id: UUID) -> dict[str, Any]:
        """Make one recorded attempt at a request with a credential fetched for this attempt.

        The credential is fetched before error handling starts, so a failure there is not recorded. A
        ``PipelineError`` or ``httpx.HTTPError`` from the exchange is recorded with ``_status`` and ``_result``; for a
        ``TokenRejected``, the credential is then passed to ``tokens.reject``; the error is re-raised. A success is
        recorded with the response status and ``"success"``.

        Preconditions:
            None.

        Args:
            request: The request to send.
            source_id: The source the request is made for.

        Returns:
            The response body from ``_call``.

        Raises:
            PipelineError: The translated error from ``_exchange``: a ``map_error`` result, a ``TransientError`` for
                a network failure, or a ``ClientErrorUnhandled`` for a 2xx body that is not a JSON object.
            httpx.HTTPError: Any ``httpx`` error that ``translates_errors`` does not translate.
            Any exception from ``tokens.get``, ``tokens.reject``, ``build`` or ``fault``, unchanged. An exception from
                ``tokens.reject`` replaces the ``TokenRejected``. A ``TransientError`` from it is retried by ``send``
                like any other; the next attempt fetches a fresh credential. The attempt is already recorded as
                ``token_rejected``, and the ``TokenRejected`` stays as the new error's ``__context__``.

        """
        # Fresh credential and fresh build (TikTok timestamp + sign) on every attempt.
        credential = await self.tokens.get(source_id, self.connector)
        try:
            status, body = await self._exchange(request, credential)
        except (PipelineError, httpx.HTTPError) as err:
            self._record(_status(err), _result(err))
            if isinstance(err, TokenRejected):
                await self.tokens.reject(source_id, self.connector, credential)
            raise
        self._record(status, "success")
        return body

    async def _call(self, request: ConnectorRequest, credential: C) -> tuple[int, dict[str, Any]]:
        """Send one request and return its status and JSON-object body, or raise the error it reports.

        A 2xx response succeeds when its body is a JSON object and ``fault`` returns ``None``. Any other response is
        raised as a ``ConnectorResponseError`` whose code, message and request id come from ``fault``, or, when the body
        is not a JSON object or ``fault`` returns ``None``, from no code, the first 500 characters of the response
        text, and no request id. Called through ``_exchange``, which translates what it raises.

        Preconditions:
            None.

        Args:
            request: The connector request to send.
            credential: The credential to build the request with.

        Returns:
            The response status code and the body parsed as a JSON object.

        Raises:
            ClientErrorUnhandled: If a 2xx response body is not a JSON object; ``code`` and ``request_id`` are
                ``None``.
            ConnectorResponseError: If ``fault`` reports an error or the response is not 2xx. Its ``headers`` are the
                response headers.
            Any exception from ``build``, ``fault`` or ``http.send``, unchanged.

        """
        response = await self.http.send(self.build(request, credential))
        body = _json_object(response)
        if response.is_success:
            if body is None:
                raise ClientErrorUnhandled(
                    connector=self.connector,
                    path=request.path,
                    status=response.status_code,
                    code=None,
                    message="response body is not a JSON object",
                    request_id=None,
                )
            fault = self.fault(response, body)
            if fault is None:
                return response.status_code, body
        else:
            fault = self.fault(response, body) if body is not None else None
            if fault is None:
                fault = Fault(code=None, message=response.text[:500], request_id=None)
        raise ConnectorResponseError(
            connector=self.connector,
            path=request.path,
            status=response.status_code,
            code=fault.code,
            message=fault.message,
            request_id=fault.request_id,
            headers=response.headers,
        )

    def _record(self, status: int, result: str) -> None:
        """Add one to the connector request counter, labelled with the connector, status and result.

        Preconditions:
            None.

        Args:
            status: HTTP status of the attempt, or ``0`` when there was no response.
            result: ``"success"``, ``"network_error"``, ``"unhandled"`` or an ``ErrorOutcome`` value.

        Returns:
            None.

        Raises:
            Any exception from ``Counter.add``, unchanged.

        """
        self.instruments.connector_requests.add(1, {"connector": self.connector, "status": status, "result": result})


def _status(err: Exception) -> int:
    """Return the HTTP status to record for an error raised by an attempt.

    Preconditions:
        None.

    Args:
        err: The error the attempt raised.

    Returns:
        ``err.status`` for a ``ClientErrorUnhandled``; otherwise the status of ``err.__cause__`` when that is a
        ``ConnectorResponseError``; otherwise ``0``, as for network errors.

    Raises:
        None.

    """
    if isinstance(err, ClientErrorUnhandled):
        return err.status
    cause = err.__cause__
    return cause.status if isinstance(cause, ConnectorResponseError) else 0  # 0: no response arrived


def _result(err: Exception) -> str:
    """Return the result label to record for an error raised by an attempt.

    Subclasses are tested before their bases: ``RateLimited`` before ``TransientError`` and ``TokenRejected`` before
    ``AuthBlocked``.

    Preconditions:
        None.

    Args:
        err: The error the attempt raised.

    Returns:
        ``"network_error"`` if ``err`` or its ``__cause__`` is an ``httpx.HTTPError``; otherwise the
        ``ErrorOutcome`` member for its type; otherwise ``"unhandled"``, as for ``ClientErrorUnhandled``.

    Raises:
        None.

    """
    # The metric's result for a failed attempt: the inverse of map_error, plus the two non-rule results.
    if isinstance(err, httpx.HTTPError) or isinstance(err.__cause__, httpx.HTTPError):
        return "network_error"
    match err:
        case RateLimited():
            return ErrorOutcome.RATE_LIMITED
        case TransientError():
            return ErrorOutcome.TRANSIENT
        case TokenRejected():
            return ErrorOutcome.TOKEN_REJECTED
        case AuthBlocked():
            return ErrorOutcome.BLOCKED
        case PermanentError():
            return ErrorOutcome.PERMANENT
        case _:
            return "unhandled"


def _json_object(response: httpx.Response) -> dict[str, Any] | None:
    """Return the response body parsed as JSON if it is a JSON object, else ``None``.

    Preconditions:
        The response body has been read, as ``http.send`` does when not streaming.

    Args:
        response: The response to parse.

    Returns:
        The parsed body if it is a JSON object; ``None`` if it is not valid JSON or is another JSON value.

    Raises:
        Nothing for a body that is not valid JSON: ``ValueError`` from ``response.json()`` is caught.

    """
    try:
        body = response.json()
    except ValueError:
        return None
    # Trust boundary: the network can send any JSON value.
    return body if isinstance(body, dict) else None
