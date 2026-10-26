"""The Cruva connector client: the error rules and the HTTP client that sends API-key requests and applies them.

Public API:
    BASE_URL: The Cruva API host every request is sent to.
    CruvaCredential: A source's Cruva shop id: non-empty printable ASCII with no whitespace.
    CRUVA_RULES: The error rules that classify Cruva error responses, tried in order.
    CruvaClient: The HTTP connector client that authenticates Cruva requests with the account-wide API key.

"""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final

import httpx
from pydantic import SecretStr

from pipeline.connectors.client import CredentialProvider, ErrorOutcome, ErrorRule, Fault, HttpConnectorClient
from pipeline.connectors.spec import ConnectorRequest
from pipeline.telemetry import Instruments

#: The Cruva API host. ``CruvaClient.build`` appends the request path to it.
BASE_URL: Final = "https://api.cruva.com"


@dataclass(frozen=True, slots=True)
class CruvaCredential:
    """Hold the Cruva shop id a request for one source is sent with.

    The shop id is non-empty and every character is printable ASCII other than whitespace (``!`` to ``~``), because
    it is sent as the ``x-shop-id`` header; a stored id that breaks this fails here, naming it, rather than in
    ``CruvaClient.build`` or ``http.send``.

    Attributes:
        shop_id (str): The source's Cruva shop id, sent in the ``x-shop-id`` header on source-scoped requests. Not
            empty; printable ASCII with no whitespace.

    """

    shop_id: str  # Cruva's own shop id, not the TikTok shop id or cipher

    def __post_init__(self) -> None:
        """Reject a shop id that is empty or could not be sent as a header.

        Preconditions:
            None.

        Returns:
            None.

        Raises:
            ValueError: If ``shop_id`` is ``""`` or contains any character outside ``!`` to ``~`` (whitespace
                anywhere, a control character, non-ASCII). The message shows the id with ``repr``.

        """
        # sent as the x-shop-id header, so only printable ASCII without spaces can be sent
        if not self.shop_id or not all("!" <= c <= "~" for c in self.shop_id):
            raise ValueError(f"Cruva shop id must be non-empty printable ASCII with no whitespace: {self.shop_id!r}")


#: The rules ``map_error`` tries, in order, for a Cruva error. Any 429 is rate-limited and any 5xx is transient. A 403
#: whose message contains "not owned by user" or "not on an active paid plan" is blocked. A 400 whose message contains
#: "Missing X-Shop-Id header", "Invalid shop_id" or "timezone" is permanent. Keywords are matched ignoring case.
#: Every other error, including any 401, 404 or 408, becomes ``ClientErrorUnhandled``. Cruva errors carry no code.
CRUVA_RULES: Final = (
    ErrorRule(outcome=ErrorOutcome.RATE_LIMITED, status=429),
    ErrorRule(outcome=ErrorOutcome.TRANSIENT, status=range(500, 600)),
    ErrorRule(outcome=ErrorOutcome.BLOCKED, status=403, keyword="not owned by user"),
    ErrorRule(outcome=ErrorOutcome.BLOCKED, status=403, keyword="not on an active paid plan"),
    ErrorRule(outcome=ErrorOutcome.PERMANENT, status=400, keyword="Missing X-Shop-Id header"),
    ErrorRule(outcome=ErrorOutcome.PERMANENT, status=400, keyword="Invalid shop_id"),
    ErrorRule(outcome=ErrorOutcome.PERMANENT, status=400, keyword="timezone"),
)


class CruvaClient(HttpConnectorClient[CruvaCredential]):
    """Send Cruva requests authenticated with the account-wide API key and classify their errors with ``CRUVA_RULES``.

    Retrying, recording and credential handling come from ``HttpConnectorClient``.

    Attributes:
        connector (str): ``"cruva"``.
        rules (tuple[ErrorRule, ...]): ``CRUVA_RULES``.

    """

    connector = "cruva"
    rules = CRUVA_RULES

    def __init__(
        self,
        http: httpx.AsyncClient,
        tokens: CredentialProvider[CruvaCredential],
        instruments: Instruments,
        api_key: SecretStr,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """Store the dependencies and the Cruva API key.

        Preconditions:
            None.

        Args:
            http: The client requests are built and sent with.
            tokens: Where credentials come from and rejected ones go.
            instruments: Telemetry instruments.
            api_key: The API key for the whole Cruva account, sent in the ``x-api-key`` header on every request.
                The caller decides where it comes from.
            sleep: Keyword-only. Awaited with each retry wait in seconds.

        Returns:
            None.

        Raises:
            None.

        """
        super().__init__(http, tokens, instruments, sleep=sleep)
        self._api_key = api_key

    def build(self, request: ConnectorRequest, credential: CruvaCredential) -> httpx.Request:
        """Build the HTTP request for a connector request.

        The API key goes in the ``x-api-key`` header and, for a source-scoped request, the credential's shop id in the
        ``x-shop-id`` header. Every request, GET included, gets ``Content-Type: application/json``. The query is the
        request's query, merged by ``httpx`` with any default params of ``http``. The body is ``request.body``
        encoded as compact JSON by ``httpx``; with no body the content is empty.

        Preconditions:
            None; ``CruvaCredential`` guarantees the shop id can be sent as a header.

        Args:
            request: The connector request to send.
            credential: The source's credential for this attempt; unused for a request that is not source-scoped.

        Returns:
            The request to ``BASE_URL`` plus ``request.path``, with the query, the headers and the JSON body.

        Raises:
            ValueError: From ``httpx`` if the body holds a NaN or infinite float, or a circular reference.
            TypeError: From ``httpx`` if the body holds a value JSON cannot represent.
            ``HttpConnectorClient`` lets each of these escape ``send`` without recording the attempt.

        """
        headers = {"x-api-key": self._api_key.get_secret_value(), "Content-Type": "application/json"}
        if request.source_scoped:
            headers["x-shop-id"] = credential.shop_id
        return self.http.build_request(
            request.method,
            f"{BASE_URL}{request.path}",
            params=dict(request.query),
            headers=headers,
            json=request.body,
        )

    def fault(self, response: httpx.Response, body: Mapping[str, Any]) -> Fault | None:
        """Return the error a Cruva error response reports in its ``error`` field, or ``None``.

        A 2xx response always succeeds, whatever its body holds.

        Preconditions:
            None.

        Args:
            response: The response received; only its success status is read.
            body: The response body parsed as a JSON object.

        Returns:
            ``None`` for a 2xx response, or for any other response whose ``error`` is missing or not a string (so
            ``HttpConnectorClient`` falls back to the response text). Otherwise a ``Fault`` with no code, the ``error``
            string as its message and no request id.

        Raises:
            None.

        """
        # Cruva reports errors by HTTP status only; a 2xx body is never read as a fault.
        if response.is_success:
            return None
        message = body.get("error")
        return Fault(code=None, message=message, request_id=None) if isinstance(message, str) else None
