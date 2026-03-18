from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse
from httpx import AsyncClient, ConnectError, RequestError, Timeout, Limits, Response as HTTPXResponse, AsyncByteStream
import random
import time
import asyncio
from app.core.config import settings
from app.core.logging_config import setup_logging
from app.services.metrics_callback_handler import MetricsCallbackHandler
from app.schemas.prompt import ChainMetadataForTracking, ChainType
import json
import fnmatch
from typing import AsyncIterator
from app.core.security import redact_headers
from app.services.tracing import trace_proxy_request

logger = setup_logging()

# HTTP success status codes — reused across the module
_SUCCESS_STATUS_CODES = frozenset(range(200, 209)) | {226}

# Cached at module level — parsed once from settings
_EXCLUDE_HEADER_PATTERNS: frozenset[str] = frozenset(
    h.strip().lower() for h in settings.proxy_exclude_headers.split(",") if h.strip()
)


class CircuitBreaker:
    """Encapsulates circuit breaker state with proper locking."""

    def __init__(
        self,
        failure_threshold: int | None = None,
        recovery_time: int | None = None,
        window_size: int | None = None,
    ):
        self._failure_threshold = failure_threshold
        self._recovery_time = recovery_time
        self._window_size = window_size
        self.is_open = False
        self.open_time: float = 0
        self.failure_timestamps: list[float] = []
        self._lock = asyncio.Lock()

    @property
    def failure_threshold(self) -> int:
        return self._failure_threshold if self._failure_threshold is not None else settings.proxy_failure_threshold

    @property
    def recovery_time(self) -> int:
        return self._recovery_time if self._recovery_time is not None else settings.proxy_recovery_time

    @property
    def window_size(self) -> int:
        return self._window_size if self._window_size is not None else settings.proxy_window_size

    async def get_status(self) -> dict:
        async with self._lock:
            return {
                "circuit_open": self.is_open,
                "circuit_open_time": self.open_time,
                "failure_timestamps": list(self.failure_timestamps),
            }

    async def check_open(self) -> bool:
        async with self._lock:
            if self.is_open and time.time() - self.open_time < self.recovery_time:
                return True
            return False

    async def record_success(self):
        async with self._lock:
            self.failure_timestamps.clear()

    async def record_failure(self) -> bool:
        """Record a failure. Returns True if circuit breaker was activated."""
        async with self._lock:
            current_time = time.time()
            self.failure_timestamps.append(current_time)
            window_start = current_time - self.window_size
            self.failure_timestamps[:] = [
                ts for ts in self.failure_timestamps if ts > window_start
            ]
            if (
                self.failure_threshold > 0
                and len(self.failure_timestamps) >= self.failure_threshold
            ):
                self.is_open = True
                self.open_time = time.time()
                return True
            return False


circuit_breaker = CircuitBreaker()


async def create_async_client():
    cert = None
    if settings.proxy_authorization_type == "CERT":
        cert = (settings.proxy_api_cert_path, settings.proxy_api_cert_key_path)
    return AsyncClient(
        cert=cert,
        timeout=Timeout(connect=settings.proxy_connect_timeout, read=settings.proxy_read_timeout, write=settings.proxy_write_timeout, pool=settings.proxy_pool_timeout),
        limits=Limits(
            max_connections=settings.max_connections,
            max_keepalive_connections=settings.max_keepalive_connections
        ),
        verify=settings.proxy_verify_ssl
    )


async def get_circuit_status():
    return await circuit_breaker.get_status()


async def exponential_backoff_retry(
    func,
    *args,
    cb: CircuitBreaker | None = None,
    max_retries: int | None = None,
    base_delay: float | None = None,
    backoff_factor: float | None = None,
    **kwargs,
) -> JSONResponse | HTTPXResponse:
    """Performs a request with exponential backoff on retryable errors."""
    _cb = cb or circuit_breaker
    _max_retries = max_retries if max_retries is not None else settings.proxy_max_retries
    _base_delay = base_delay if base_delay is not None else settings.proxy_base_delay
    _backoff_factor = backoff_factor if backoff_factor is not None else settings.proxy_backoff_factor

    last_response_status_code = 523
    error_response = ""

    if await _cb.check_open():
        return JSONResponse(content={"error": "Circuit breaker open. Try later."}, status_code=503)

    attempts = 1 + _max_retries
    for attempt in range(attempts):
        try:
            response = await func(*args, **kwargs)
            last_response_status_code = response.status_code
            error_response = extract_content(response)
            # Handle 429 (Too Many Requests) with Retry-After
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                delay = int(retry_after) if retry_after else _base_delay * (_backoff_factor ** attempt) + random.uniform(0, 0.1)
                logger.warning(f"Rate limited. Retrying in {delay:.2f} seconds.")
                await asyncio.sleep(delay)
                continue

            if response.status_code not in {429, 500, 502, 503, 504}:
                await _cb.record_success()
                return response

        except ConnectError as e:
            error_response = {
                "error": {
                    "message": "Connection failed",
                    "details": {
                        "exception": str(e),
                        "exception_type": type(e).__name__,
                        "attempt": attempt + 1
                    }
                }
            }
            logger.warning("Connection failed", details=error_response)
        except RequestError as e:
            error_response = {
                "error": {
                    "message": "Request failed",
                    "details": {
                        "exception": str(e),
                        "exception_type": type(e).__name__,
                        "attempt": attempt + 1
                    }
                }
            }
            logger.warning("Request failed", details=error_response)

        activated = await _cb.record_failure()
        if activated:
            logger.error("Circuit breaker activated due to multiple failures.", details=error_response)
            return JSONResponse(content={"error": "Circuit breaker activated. Try later."}, status_code=503)

        # Exponential backoff delay
        delay = _base_delay * (_backoff_factor ** attempt) + random.uniform(0, 0.1)
        logger.debug(f"Retrying request in {delay:.2f} seconds.")
        await asyncio.sleep(delay)

    logger.error("Max retries exceeded", details=error_response)
    return JSONResponse(content=error_response, status_code=last_response_status_code)


class RequestStreamWrapper(AsyncByteStream):
    def __init__(self, request):
        self._stream = request.stream()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._stream:
            yield chunk


async def stream_multipart_post(request: Request, client: AsyncClient, target_url: str, headers: dict) -> JSONResponse:
    forwarded_headers = {
        key: value
        for key, value in headers.items()
        if key.lower() not in ("content-length", "transfer-encoding", "connection", "expect", "host")
    }

    async with client.stream(
            method="POST",
            url=target_url,
            headers=forwarded_headers,
            content=RequestStreamWrapper(request),
    ) as response:
        body = await response.aread()
        try:
            json_data = extract_content(response, True)
        except Exception:
            try:
                json_data = json.loads(body.decode("utf-8", errors="replace"))
            except Exception:
                json_data = {"error": "Invalid JSON response"}

        return JSONResponse(content=json_data, status_code=response.status_code)


def extract_content(response, raiseException=False) -> dict:
    if isinstance(response, HTTPXResponse):
        try:
            return response.json()
        except Exception:
            try:
                return {"detail": response.text}
            except Exception as e:
                if raiseException:
                    raise
                return {"detail": f"Unable to parse httpx response: {str(e)}"}

    if isinstance(response, JSONResponse):
        try:
            if not getattr(response, "body", None):
                response.render()
            return json.loads(response.body.decode("utf-8", errors="replace"))
        except Exception as e:
            if raiseException:
                raise
            return {"detail": f"Unable to parse JSONResponse: {str(e)}"}
    return {"detail": "Unknown response object type"}


def _parse_model_version(model_string: str) -> dict | None:
    """Parse model name:version string into version dict. Returns None on invalid format."""
    parts = model_string.split(":", 1)
    if len(parts) != 2:
        return None
    name, version = parts
    if name.endswith("-Pro"):
        name = name.replace("-Pro", "-90b-128k-base")
    elif name.endswith("-Max"):
        name = name.replace("-Max", "-38b-128k-base")
    else:
        name = f"{name}-9b-128k-base"
    return {"version": f"{name}:{version}"}


async def get_model_version(model_name: str, client: AsyncClient, request: Request, custom_headers: dict[str, str] | None = None):
    custom_headers = custom_headers or {}
    path = "v1/chat/completions"
    target_url = f"{settings.proxy_target_url}/{path}"
    body = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Reply with any single digit"}],
        "stream": False,
        "update_interval": 0
    }
    try:
        response = await exponential_backoff_retry(
            client.request, "POST", target_url, headers=custom_headers, json=body
        )
        if isinstance(response, HTTPXResponse) and response.status_code in _SUCCESS_STATUS_CODES:
            current_model = response.json().get("model")
            if current_model:
                result = _parse_model_version(current_model)
                if result:
                    return result
                logger.warning(f"Unexpected model format: {current_model}")
        else:
            response_dict = extract_content(response)
            logger.error("Proxy error details", details={
                "status_code": response.status_code,
                "target_url": target_url,
                "response": response_dict,
            })
            error_response = {
                "error": {
                    "status_code": response.status_code,
                    "message": response_dict.get("message", "Proxy request failed"),
                    "details": {
                        "response": response_dict
                    }
                }
            }
            return JSONResponse(content=error_response, status_code=response.status_code)

    except Exception as e:
        logger.error("get_model_version exception", details={
            "target_url": target_url,
            "exception": str(e),
            "exception_type": type(e).__name__
        })
        return JSONResponse(content={
            "error": {
                "status_code": 500,
                "message": "get_model_version request failed",
            }
        }, status_code=500)


# ---------------------------------------------------------------------------
# Shared proxy helpers (used by both legacy and multi-provider paths)
# ---------------------------------------------------------------------------

def _prepare_proxy_headers(request: Request, auth_headers: dict[str, str]) -> dict[str, str]:
    """Build forwarded headers: filter excluded patterns, apply auth, force identity encoding."""
    headers = dict(request.headers)
    logger.debug(f"original headers {redact_headers(headers)}")
    headers = {
        k: v for k, v in headers.items()
        if not any(fnmatch.fnmatch(k.lower(), p) for p in _EXCLUDE_HEADER_PATTERNS)
    }
    logger.debug(f"cleaned headers {redact_headers(headers)}")
    headers.update(auth_headers)
    headers["accept-encoding"] = "identity"
    logger.debug(f"final headers {redact_headers(headers)}")
    return headers


def _detect_streaming(method: str, body: bytes) -> bool:
    """Return True if request body indicates SSE streaming."""
    if method.lower() == "post" and body:
        try:
            return json.loads(body).get("stream") is True
        except (json.JSONDecodeError, AttributeError):
            pass
    return False


def _emit_completions_metrics(
    response_data: dict,
    request: Request,
    start_time: float,
    body: bytes,
    path: str,
    provider_prefix: str | None,
    currency: str | None,
    pricing_cache,
) -> None:
    """Emit Prometheus metrics and tracing for a non-streaming completions response."""
    if "completions" not in path:
        return

    metadata = ChainMetadataForTracking(
        chain_type=ChainType.prompt,
        chain_name="proxy",
        group_id=request.headers.get("x-group-id", "unknown"),
    )

    # Resolve pricing if not already known
    pfx, cur = provider_prefix, currency
    if not pfx and pricing_cache:
        model_name = response_data.get("model", "")
        usage = response_data.get("usage", {})
        found = pricing_cache.find_cost(
            model_name,
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )
        if found:
            pfx, cur, _ = found

    try:
        MetricsCallbackHandler(
            metadata,
            provider_prefix=pfx,
            currency=cur,
            pricing_cache=pricing_cache,
        ).on_llm_end(response_data)
    except Exception as e:
        logger.error("Error processing LLM usage metrics", exc_info=e)

    duration_ms = (time.time() - start_time) * 1000
    try:
        input_body = json.loads(body) if body else None
    except (json.JSONDecodeError, AttributeError):
        input_body = None
    trace_proxy_request(
        model=response_data.get("model", ""),
        provider=pfx,
        input_body=input_body,
        output_body=response_data,
        status_code=200,
        usage=response_data.get("usage"),
        duration_ms=duration_ms,
        group_id=request.headers.get("x-group-id", "unknown"),
        is_streaming=False,
    )


def _make_error_response(response, provider_label: str | None = None) -> JSONResponse:
    """Build a standardised error JSONResponse from a failed upstream response."""
    response_dict = extract_content(response)
    log_details: dict = {
        "status_code": response.status_code,
        "response": response_dict,
    }
    if provider_label:
        log_details["provider"] = provider_label
    logger.error("Proxy error details", details=log_details)
    return JSONResponse(
        content={
            "error": {
                "status_code": response.status_code,
                "message": response_dict.get("message", "Proxy request failed"),
                "details": {"response": response_dict},
            }
        },
        status_code=response.status_code,
    )


async def _do_proxy_request(
    client: AsyncClient,
    target_url: str,
    method: str,
    headers: dict[str, str],
    body: bytes,
    path: str,
    request: Request,
    provider_prefix: str | None = None,
    provider_currency: str | None = None,
    pricing_cache=None,
    cb: CircuitBreaker | None = None,
    max_retries: int | None = None,
    base_delay: float | None = None,
    backoff_factor: float | None = None,
) -> JSONResponse | StreamingResponse:
    """Core proxy logic shared by legacy single-provider and multi-provider paths."""
    is_streaming = _detect_streaming(method, body)

    if is_streaming:
        return await _handle_streaming_request(
            client=client,
            target_url=target_url,
            headers=headers,
            body=body,
            path=path,
            request=request,
            provider_prefix=provider_prefix,
            provider_currency=provider_currency,
            pricing_cache=pricing_cache,
        )

    start_time = time.time()
    response = await exponential_backoff_retry(
        client.request,
        method,
        target_url,
        headers=headers,
        content=body,
        cb=cb,
        max_retries=max_retries,
        base_delay=base_delay,
        backoff_factor=backoff_factor,
    )

    if isinstance(response, HTTPXResponse) and response.status_code in _SUCCESS_STATUS_CODES:
        logger.debug(f"Proxy request successful: {method} {target_url} -> {response.status_code}")

        if "completions" in path:
            response_data = json.loads(response.text)
            _emit_completions_metrics(
                response_data, request, start_time, body, path,
                provider_prefix, provider_currency, pricing_cache,
            )

        return JSONResponse(
            content=extract_content(response),
            status_code=response.status_code,
        )

    return _make_error_response(response, provider_label=provider_prefix)


# ---------------------------------------------------------------------------
# Public entry-points (kept for backward-compat with app.py and tests)
# ---------------------------------------------------------------------------

async def proxy_request_with_retries(client: AsyncClient, path: str, request: Request, custom_headers: dict[str, str] | None = None, pricing_cache=None):
    custom_headers = custom_headers or {}
    target_url = f"{settings.proxy_target_url}/{path}"
    if request.url.query:
        target_url = f"{target_url}?{request.url.query}"

    headers = _prepare_proxy_headers(request, custom_headers)

    try:
        method = request.method
        if method.lower() == "post" and headers.get("content-type", "").lower().startswith("multipart/form-data"):
            return await stream_multipart_post(request, client, target_url, headers=headers)

        body = await request.body()

        return await _do_proxy_request(
            client=client,
            target_url=target_url,
            method=method,
            headers=headers,
            body=body,
            path=path,
            request=request,
            pricing_cache=pricing_cache,
        )

    except Exception as e:
        logger.error("Proxy exception", details={
            "target_url": target_url,
            "method": request.method,
            "exception": str(e),
            "exception_type": type(e).__name__
        })
        return JSONResponse(content={
            "error": {
                "status_code": 500,
                "message": "Proxy request failed",
            }
        }, status_code=500)


async def _handle_streaming_request(
    client: AsyncClient, target_url: str, headers: dict,
    body: bytes, path: str, request: Request,
    provider_prefix: str | None = None,
    provider_currency: str | None = None,
    pricing_cache=None,
) -> StreamingResponse | JSONResponse:
    """Handle a streaming proxy request, forwarding SSE chunks from upstream."""
    start_time = time.time()
    try:
        upstream_response = await client.send(
            client.build_request("POST", target_url, headers=headers, content=body),
            stream=True,
        )
    except (ConnectError, RequestError) as e:
        logger.error("Streaming connection failed", details={
            "target_url": target_url,
            "exception": str(e),
            "exception_type": type(e).__name__,
        })
        return JSONResponse(
            content={"error": {"status_code": 502, "message": "Upstream connection failed"}},
            status_code=502,
        )

    if upstream_response.status_code not in _SUCCESS_STATUS_CODES:
        body_bytes = await upstream_response.aread()
        await upstream_response.aclose()
        try:
            error_data = json.loads(body_bytes.decode("utf-8", errors="replace"))
        except Exception:
            error_data = {"detail": body_bytes.decode("utf-8", errors="replace")}
        return JSONResponse(content=error_data, status_code=upstream_response.status_code)

    async def _stream_generator():
        collected_chunks: list[str] = []
        try:
            async for chunk in upstream_response.aiter_bytes():
                collected_chunks.append(chunk.decode("utf-8", errors="replace"))
                yield chunk
        finally:
            await upstream_response.aclose()
            # Parse collected SSE data for metrics
            if "completions" in path:
                _emit_streaming_metrics(
                    collected_chunks, request, start_time, body,
                    provider_prefix, provider_currency, pricing_cache,
                )

    return StreamingResponse(
        _stream_generator(),
        status_code=upstream_response.status_code,
        media_type="text/event-stream",
    )


def _emit_streaming_metrics(
    chunks: list[str],
    request: Request,
    start_time: float = 0,
    body: bytes = b"",
    provider_prefix: str | None = None,
    provider_currency: str | None = None,
    pricing_cache=None,
) -> None:
    """Parse SSE chunks for usage data and emit metrics + tracing."""
    try:
        full_text = "".join(chunks)
        # SSE lines look like: "data: {...}\n\n" — find the last data line before [DONE]
        last_data = None
        for line in full_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("data:") and stripped != "data: [DONE]":
                last_data = stripped[len("data:"):].strip()

        if last_data:
            last_payload = json.loads(last_data)
            if "usage" in last_payload:
                metadata = ChainMetadataForTracking(
                    chain_type=ChainType.prompt,
                    chain_name="proxy",
                    group_id=request.headers.get("x-group-id", "unknown"),
                )
                # Resolve pricing from any provider if not set
                pfx, cur = provider_prefix, provider_currency
                if not pfx and pricing_cache:
                    usage = last_payload.get("usage", {})
                    found = pricing_cache.find_cost(
                        last_payload.get("model", ""),
                        usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0),
                    )
                    if found:
                        pfx, cur, _ = found

                MetricsCallbackHandler(
                    metadata,
                    provider_prefix=pfx,
                    currency=cur,
                    pricing_cache=pricing_cache,
                ).on_llm_end(last_payload)

            duration_ms = (time.time() - start_time) * 1000 if start_time else 0
            try:
                input_body = json.loads(body) if body else None
            except (json.JSONDecodeError, AttributeError):
                input_body = None
            trace_proxy_request(
                model=last_payload.get("model", ""),
                provider=provider_prefix,
                input_body=input_body,
                output_body=last_payload,
                status_code=200,
                usage=last_payload.get("usage"),
                duration_ms=duration_ms,
                group_id=request.headers.get("x-group-id", "unknown"),
                is_streaming=True,
            )
    except Exception as e:
        logger.error("Error processing streaming LLM usage metrics", exc_info=e)


def _strip_model_prefix(body: bytes, original_model: str, stripped_model: str) -> bytes:
    """Replace the prefixed model name in the request body with the stripped name."""
    try:
        body_json = json.loads(body)
        if body_json.get("model") == original_model:
            body_json["model"] = stripped_model
            return json.dumps(body_json).encode("utf-8")
    except (json.JSONDecodeError, AttributeError):
        pass
    return body


async def proxy_request_to_provider(
    provider,
    path: str,
    request: Request,
    auth_headers: dict[str, str],
    original_model: str,
    stripped_model: str,
    pricing_cache=None,
):
    """Route a request to a specific LLM provider, stripping the model prefix."""
    config = provider.config
    target_url = f"{config.base_url}/{path}"
    if request.url.query:
        target_url = f"{target_url}?{request.url.query}"

    headers = _prepare_proxy_headers(request, auth_headers)

    try:
        body = await request.body()
        body = _strip_model_prefix(body, original_model, stripped_model)

        return await _do_proxy_request(
            client=provider.client,
            target_url=target_url,
            method=request.method,
            headers=headers,
            body=body,
            path=path,
            request=request,
            provider_prefix=config.prefix,
            provider_currency=config.currency,
            pricing_cache=pricing_cache,
            cb=provider.circuit_breaker,
            max_retries=config.max_retries,
            base_delay=config.base_delay,
            backoff_factor=config.backoff_factor,
        )

    except Exception as e:
        logger.error("Provider proxy exception", details={
            "provider": config.prefix,
            "target_url": target_url,
            "exception": str(e),
            "exception_type": type(e).__name__,
        })
        return JSONResponse(
            content={"error": {"status_code": 500, "message": "Proxy request failed"}},
            status_code=500,
        )
