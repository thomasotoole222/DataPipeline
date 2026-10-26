"""The TikTok Shop connector client: request signing, error rules and the HTTP client that applies them.

Public API:
    BASE_URL: The TikTok Shop Open API host every request is sent to.
    CONNECTOR: The connector name, ``"tiktok_shop"``.
    TikTokShopCredential: A source's non-empty access token and shop cipher.
    encode_body: Serialise a request body to the compact JSON text that is both signed and sent.
    sign: Compute the HMAC-SHA256 signature of a request's path, query and body.
    TIKTOK_SHOP_RULES: The error rules that classify TikTok Shop error responses, tried in order.
    TikTokShopClient: The HTTP connector client that signs TikTok Shop requests and reads their error codes.

"""

import asyncio
import hashlib
import hmac
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

import httpx
from pydantic import SecretStr

from pipeline.connectors.client import CredentialProvider, ErrorOutcome, ErrorRule, Fault, HttpConnectorClient
from pipeline.connectors.spec import ConnectorRequest
from pipeline.telemetry import Instruments
from pipeline.timeutil import to_epoch, utc_now

#: The TikTok Shop Open API host. ``TikTokShopClient.build`` appends the request path to it.
BASE_URL: Final = "https://open-api.tiktokglobalshop.com"
#: The connector name, used as ``TikTokShopClient.connector`` in errors, telemetry and credential lookups.
CONNECTOR: Final = "tiktok_shop"

# Query keys the client owns; a request that already carries one is a spec bug.
#: Query parameters the client sets itself; ``TikTokShopClient.build`` rejects a request whose query sets any of
#: them. ``access_token`` is listed although the client sends the token in a header, not the query.
_RESERVED: Final = frozenset({"app_key", "timestamp", "shop_cipher", "sign", "access_token"})
# Not covered by the signature.
#: Query parameters ``sign`` leaves out of the signed text.
_UNSIGNED: Final = frozenset({"sign", "access_token"})
#: The message keyword that marks a 36009004 error as a rejected token rather than a permanent error.
_TOKEN_KEYWORD: Final = "access_token"  # noqa: S105 - a message keyword, not a secret


@dataclass(frozen=True, slots=True)
class TikTokShopCredential:
    """Hold the access token and shop cipher a TikTok Shop request for one source is sent with.

    ``__post_init__`` enforces that neither value is empty. Whitespace-only values are accepted.

    Attributes:
        access_token (SecretStr): The source's access token, sent in the ``x-tts-access-token`` header. Not empty.
        shop_cipher (str): The source's shop cipher, sent as the ``shop_cipher`` query parameter on source-scoped
            requests. Not empty.

    """

    access_token: SecretStr
    shop_cipher: str

    def __post_init__(self) -> None:
        """Reject an empty access token or shop cipher.

        Preconditions:
            None.

        Returns:
            None.

        Raises:
            ValueError: If the access token's secret value is ``""``, or ``shop_cipher`` is ``""``.

        """
        if not self.access_token.get_secret_value():
            raise ValueError("access_token must not be empty")
        if not self.shop_cipher:
            raise ValueError("shop_cipher must not be empty")


def encode_body(body: Mapping[str, Any] | None) -> str:
    """Serialise a request body to compact JSON text, or ``""`` for no body.

    The text has no spaces after separators and keeps non-ASCII characters unescaped. Key order is the mapping's
    own order, not sorted.

    Preconditions:
        None.

    Args:
        body: The JSON body, or ``None``.

    Returns:
        ``""`` when ``body`` is ``None``; otherwise its JSON text. An empty mapping gives ``"{}"``.

    Raises:
        ValueError: From ``json.dumps`` if the body holds a NaN or infinite float, or a circular reference.
        TypeError: From ``json.dumps`` if the body holds a value JSON cannot represent.

    """
    if body is None:
        return ""
    # Compact and stable: the signature covers these exact characters, and they are the bytes sent.
    return json.dumps(body, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sign(*, secret: str, path: str, query: Mapping[str, str], body: str) -> str:
    """Return the HMAC-SHA256 signature of a request, as TikTok Shop's ``sign`` parameter expects.

    The signed text is ``secret + path + params + body + secret``, where ``params`` is each query key followed by its
    value, keys sorted by code point, ``sign`` and ``access_token`` left out. The HMAC key is ``secret``, encoded
    as UTF-8.

    Preconditions:
        None.

    Args:
        secret: Keyword-only. The app secret.
        path: Keyword-only. The request path, exactly as sent.
        query: Keyword-only. The query parameters to sign.
        body: Keyword-only. The body text exactly as sent, ``""`` for none.

    Returns:
        The signature as 64 lowercase hex characters.

    Raises:
        UnicodeEncodeError: If ``secret`` or the signed text contains a lone surrogate.

    """
    params = "".join(f"{key}{query[key]}" for key in sorted(query) if key not in _UNSIGNED)
    wrapped = f"{secret}{path}{params}{body}{secret}"
    return hmac.new(secret.encode(), wrapped.encode(), hashlib.sha256).hexdigest()


def _codes(outcome: ErrorOutcome, *codes: int) -> tuple[ErrorRule, ...]:
    """Build one code-only rule per error code, all with the same outcome.

    Preconditions:
        None.

    Args:
        outcome: The outcome every rule names.
        *codes: The TikTok Shop error codes to match.

    Returns:
        One ``ErrorRule`` per code, in the order given; empty when no codes are given.

    Raises:
        None. Each rule sets ``code``, so ``ErrorRule.__post_init__`` accepts it.

    """
    return tuple(ErrorRule(outcome=outcome, code=code) for code in codes)


#: The rules ``map_error`` tries, in order, for a TikTok Shop error. Any 429 is rate-limited and any 408 or 5xx is
#: transient, whatever the code. Code 36009004 is a rejected token when its message contains ``access_token``,
#: otherwise permanent. Errors with no code: 401 is a rejected token and any other 4xx is permanent. A coded error
#: that no rule names becomes ``ClientErrorUnhandled``, whatever its HTTP status; TikTok often reports errors on a
#: 200, and those match the code rules like any other.
TIKTOK_SHOP_RULES: Final[tuple[ErrorRule, ...]] = (
    ErrorRule(outcome=ErrorOutcome.RATE_LIMITED, status=429),
    *_codes(ErrorOutcome.RATE_LIMITED, 36009002),
    *_codes(ErrorOutcome.TRANSIENT, 36009003, 36009007),
    ErrorRule(outcome=ErrorOutcome.TRANSIENT, status=408),
    ErrorRule(outcome=ErrorOutcome.TRANSIENT, status=range(500, 600)),
    *_codes(ErrorOutcome.TOKEN_REJECTED, 105002, 101000),
    # Must precede the PERMANENT 36009004 rule: first match wins.
    ErrorRule(outcome=ErrorOutcome.TOKEN_REJECTED, code=36009004, keyword=_TOKEN_KEYWORD),
    *_codes(ErrorOutcome.BLOCKED, 105005, 36009033),
    *_codes(
        ErrorOutcome.PERMANENT,
        106001,
        36009004,
        36009009,
        36009010,
        36009014,
        36009021,
        36009022,
        36009023,
        106013,
    ),
    # No readable code: fall back on the HTTP status (429, 408 and 5xx are covered above).
    ErrorRule(outcome=ErrorOutcome.TOKEN_REJECTED, status=401, codeless=True),
    ErrorRule(outcome=ErrorOutcome.PERMANENT, status=range(400, 500), codeless=True),
)


class TikTokShopClient(HttpConnectorClient[TikTokShopCredential]):
    """Send signed TikTok Shop requests and classify their errors with ``TIKTOK_SHOP_RULES``.

    Each attempt is rebuilt by ``build``, so it carries a fresh timestamp and signature. Retrying, recording and
    credential handling come from ``HttpConnectorClient``.

    Attributes:
        connector (str): ``CONNECTOR``.
        rules (tuple[ErrorRule, ...]): ``TIKTOK_SHOP_RULES``.

    """

    connector = CONNECTOR
    rules = TIKTOK_SHOP_RULES

    def __init__(
        self,
        http: httpx.AsyncClient,
        tokens: CredentialProvider[TikTokShopCredential],
        instruments: Instruments,
        app_key: SecretStr,
        app_secret: SecretStr,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        """Store the dependencies, the app key and secret, and the clock used for request timestamps.

        Preconditions:
            None.

        Args:
            http: The client requests are built and sent with.
            tokens: Where credentials come from and rejected ones go.
            instruments: Telemetry instruments.
            app_key: The TikTok Shop app key, sent as the ``app_key`` query parameter and signed. The caller decides
                where it comes from.
            app_secret: The TikTok Shop app secret that ``sign`` keys the signature with; never sent.
            sleep: Keyword-only. Awaited with each retry wait in seconds.
            clock: Keyword-only. Returns the current time for the ``timestamp`` parameter. Defaults to ``utc_now``.

        Returns:
            None.

        Raises:
            ValueError: If ``http`` has default query params, which ``httpx`` would merge into the URL after
                ``build`` has signed the query.

        """
        # httpx merges a client's default params into the URL after build has signed the query
        if http.params:
            raise ValueError(f"the TikTok Shop http client must have no default params: {sorted(http.params.keys())}")
        super().__init__(http, tokens, instruments, sleep=sleep)
        self._app_key = app_key
        self._app_secret = app_secret
        self._clock = clock

    def build(self, request: ConnectorRequest, credential: TikTokShopCredential) -> httpx.Request:
        """Build the signed HTTP request for a connector request.

        The query is the request's query plus ``app_key``, ``timestamp`` (whole epoch seconds from the clock) and, for
        a source-scoped request, ``shop_cipher``; it is signed with ``sign`` over the request path and the body text
        from ``encode_body``, and the signature is added as ``sign``. The access token goes in the
        ``x-tts-access-token`` header. A POST also gets ``content-type: application/json``, even with no body, in
        which case the content is empty.

        Preconditions:
            ``clock`` returns a timezone-aware ``datetime``; the default ``utc_now`` does. ``__init__`` has rejected an
            ``http`` client with default query params, so the sent query is exactly the signed one.

        Args:
            request: The connector request to send.
            credential: The source's credential for this attempt.

        Returns:
            The request to ``BASE_URL`` plus ``request.path``, with the signed query, the headers and the body text
            encoded as UTF-8.

        Raises:
            ValueError: If the request's query sets any of ``app_key``, ``timestamp``, ``shop_cipher``, ``sign`` or
                ``access_token``; from ``to_epoch`` if the clock returns a naive ``datetime``; or from
                ``encode_body``.
            TypeError: From ``encode_body``, or from ``to_epoch`` if the clock does not return a ``datetime``.
            Any exception from ``clock``, unchanged. ``HttpConnectorClient`` lets each of these escape ``send``
                without recording the attempt.

        """
        reserved = _RESERVED & request.query.keys()
        if reserved:
            raise ValueError(f"query sets client-owned parameters: {sorted(reserved)}")
        query = {
            **request.query,
            "app_key": self._app_key.get_secret_value(),
            "timestamp": str(to_epoch(self._clock())),
        }
        if request.source_scoped:  # an account-wide endpoint rejects a shop_cipher (36009004)
            query["shop_cipher"] = credential.shop_cipher
        body = encode_body(request.body)
        query["sign"] = sign(secret=self._app_secret.get_secret_value(), path=request.path, query=query, body=body)
        headers = {"x-tts-access-token": credential.access_token.get_secret_value()}
        if request.method != "GET":
            headers["content-type"] = "application/json"
        return self.http.build_request(
            request.method, BASE_URL + request.path, params=query, headers=headers, content=body.encode()
        )

    def fault(self, response: httpx.Response, body: Mapping[str, Any]) -> Fault | None:
        """Return the error a TikTok Shop response body reports, or ``None`` if it reports none.

        The body's ``code`` counts only when it is an ``int`` and not a ``bool``. ``message`` and ``request_id`` are
        used only when they are strings; otherwise the message is ``""`` and the request id ``None``.

        Preconditions:
            None.

        Args:
            response: The response received; only its success status is read.
            body: The response body parsed as a JSON object.

        Returns:
            ``None`` when ``code`` is ``0``, whatever the status, or when there is no integer ``code`` and the response
            is not 2xx (so ``HttpConnectorClient`` falls back to the response text). For a 2xx response with no integer
            ``code``, a ``Fault`` with no code and the message ``"response has no integer code"``. Otherwise a
            ``Fault`` with the code, message and request id.

        Raises:
            None.

        """
        code = body.get("code")
        message = body.get("message")
        request_id = body.get("request_id")
        # Trust boundary: the envelope's fields come from the network.
        text = message if isinstance(message, str) else ""
        rid = request_id if isinstance(request_id, str) else None
        if not isinstance(code, int) or isinstance(code, bool):
            # An error status without an envelope: the base falls back on the response text.
            if not response.is_success:
                return None
            return Fault(code=None, message="response has no integer code", request_id=rid)
        if code == 0:
            return None
        return Fault(code=code, message=text, request_id=rid)
