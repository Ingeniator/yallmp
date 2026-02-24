import pytest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import Response as HTTPXResponse, Request as HTTPXRequest, AsyncClient
from fastapi.responses import JSONResponse
from starlette.requests import Request

import app.core.proxy as proxy_mod
from app.core.proxy import (
    _parse_model_version,
    get_model_version,
    proxy_request_with_retries,
    stream_multipart_post,
    RequestStreamWrapper,
    create_async_client,
    get_circuit_status,
    CircuitBreaker,
)


# --- _parse_model_version ---


def test_parse_model_version_pro():
    result = _parse_model_version("GigaChat-Pro:1.0")
    assert result == {"version": "GigaChat-90b-128k-base:1.0"}


def test_parse_model_version_max():
    result = _parse_model_version("GigaChat-Max:2.0")
    assert result == {"version": "GigaChat-38b-128k-base:2.0"}


def test_parse_model_version_default():
    result = _parse_model_version("GigaChat:3.0")
    assert result == {"version": "GigaChat-9b-128k-base:3.0"}


def test_parse_model_version_invalid():
    assert _parse_model_version("no-colon-here") is None


# --- get_circuit_status ---


@pytest.fixture(autouse=True)
def _reset_circuit():
    proxy_mod.circuit_breaker = CircuitBreaker()


@pytest.mark.asyncio
async def test_get_circuit_status():
    result = await get_circuit_status()
    assert result["circuit_open"] is False
    assert result["circuit_open_time"] == 0
    assert result["failure_timestamps"] == []


# --- create_async_client ---


@pytest.mark.asyncio
async def test_create_async_client_default():
    with patch.object(proxy_mod, "settings") as s:
        s.proxy_authorization_type = "BEARER"
        s.proxy_connect_timeout = 10
        s.proxy_read_timeout = 300
        s.proxy_write_timeout = 30
        s.proxy_pool_timeout = None
        s.max_connections = 100
        s.max_keepalive_connections = 20
        s.proxy_verify_ssl = False

        client = await create_async_client()

    assert isinstance(client, AsyncClient)
    await client.aclose()


@pytest.mark.asyncio
async def test_create_async_client_cert_branch():
    """Verify the CERT branch sets the cert tuple (mocked to avoid real SSL)."""
    with patch.object(proxy_mod, "settings") as s, \
         patch("app.core.proxy.AsyncClient") as MockClient:
        s.proxy_authorization_type = "CERT"
        s.proxy_api_cert_path = "/path/cert.pem"
        s.proxy_api_cert_key_path = "/path/key.pem"
        s.proxy_connect_timeout = 10
        s.proxy_read_timeout = 300
        s.proxy_write_timeout = 30
        s.proxy_pool_timeout = None
        s.max_connections = 100
        s.max_keepalive_connections = 20
        s.proxy_verify_ssl = False

        await create_async_client()

    call_kwargs = MockClient.call_args[1]
    assert call_kwargs["cert"] == ("/path/cert.pem", "/path/key.pem")


# --- get_model_version ---


def _fake_request():
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/llm/v1/chat/completions",
        "query_string": b"",
        "headers": [],
        "root_path": "",
    }
    return Request(scope, receive=AsyncMock(return_value={"type": "http.request", "body": b""}))


@pytest.mark.asyncio
async def test_get_model_version_success():
    mock_response = HTTPXResponse(
        status_code=200,
        json={"model": "GigaChat:3.0"},
        request=HTTPXRequest("POST", "http://upstream/v1/chat/completions"),
    )

    with patch.object(proxy_mod, "exponential_backoff_retry", AsyncMock(return_value=mock_response)), \
         patch.object(proxy_mod, "settings") as s:
        s.proxy_target_url = "http://upstream"

        result = await get_model_version("GigaChat", AsyncMock(), _fake_request())

    assert result == {"version": "GigaChat-9b-128k-base:3.0"}


@pytest.mark.asyncio
async def test_get_model_version_error_response():
    mock_response = HTTPXResponse(
        status_code=400,
        json={"message": "bad request"},
        request=HTTPXRequest("POST", "http://upstream/v1/chat/completions"),
    )

    with patch.object(proxy_mod, "exponential_backoff_retry", AsyncMock(return_value=mock_response)), \
         patch.object(proxy_mod, "settings") as s:
        s.proxy_target_url = "http://upstream"

        result = await get_model_version("GigaChat", AsyncMock(), _fake_request())

    assert isinstance(result, JSONResponse)
    assert result.status_code == 400


@pytest.mark.asyncio
async def test_get_model_version_json_response_passthrough():
    """When exponential_backoff_retry returns a JSONResponse (e.g., circuit breaker 503), pass it through."""
    mock_response = JSONResponse(content={"error": "Circuit breaker open"}, status_code=503)

    with patch.object(proxy_mod, "exponential_backoff_retry", AsyncMock(return_value=mock_response)), \
         patch.object(proxy_mod, "settings") as s:
        s.proxy_target_url = "http://upstream"

        result = await get_model_version("GigaChat", AsyncMock(), _fake_request())

    assert isinstance(result, JSONResponse)
    assert result.status_code == 503


@pytest.mark.asyncio
async def test_get_model_version_exception():
    with patch.object(proxy_mod, "exponential_backoff_retry", AsyncMock(side_effect=RuntimeError("boom"))), \
         patch.object(proxy_mod, "settings") as s:
        s.proxy_target_url = "http://upstream"

        result = await get_model_version("GigaChat", AsyncMock(), _fake_request())

    assert isinstance(result, JSONResponse)
    assert result.status_code == 500


# --- proxy_request_with_retries ---


def _make_starlette_request(method="GET", path="/llm/v1/chat/completions", query_string=b""):
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "query_string": query_string,
        "headers": [
            (b"accept", b"application/json"),
            (b"content-type", b"application/json"),
        ],
        "root_path": "",
    }
    return Request(scope, receive=AsyncMock(return_value={"type": "http.request", "body": b"{}"}))


def _patch_settings():
    return patch.object(proxy_mod, "settings", **{
        "proxy_target_url": "http://upstream",
        "proxy_exclude_headers": "host,authorization",
        "proxy_max_retries": 0,
        "proxy_base_delay": 0,
        "proxy_backoff_factor": 1,
        "proxy_failure_threshold": 0,
        "proxy_window_size": 60,
        "proxy_recovery_time": 30,
    })


@pytest.mark.asyncio
async def test_proxy_request_completions_success():
    """Success path with 'completions' in path triggers metrics recording."""
    mock_response = HTTPXResponse(
        status_code=200,
        json={"model": "test", "usage": {"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5}},
        request=HTTPXRequest("POST", "http://upstream/v1/chat/completions"),
    )
    mock_client = AsyncMock(spec=AsyncClient)
    mock_client.request = AsyncMock(return_value=mock_response)

    with _patch_settings():
        result = await proxy_request_with_retries(
            mock_client, "v1/chat/completions",
            _make_starlette_request("POST"), {},
        )

    assert isinstance(result, JSONResponse)
    assert result.status_code == 200


@pytest.mark.asyncio
async def test_proxy_request_error_response():
    """Non-success status code returns error envelope."""
    mock_response = HTTPXResponse(
        status_code=422,
        json={"message": "validation error"},
        request=HTTPXRequest("POST", "http://upstream/v1/chat/completions"),
    )
    mock_client = AsyncMock(spec=AsyncClient)
    mock_client.request = AsyncMock(return_value=mock_response)

    with _patch_settings():
        result = await proxy_request_with_retries(
            mock_client, "v1/chat/completions",
            _make_starlette_request("POST"), {},
        )

    assert isinstance(result, JSONResponse)
    assert result.status_code == 422


@pytest.mark.asyncio
async def test_proxy_request_exception():
    """Generic exception returns 500."""
    mock_client = AsyncMock(spec=AsyncClient)
    mock_client.request = AsyncMock(side_effect=RuntimeError("unexpected"))

    with _patch_settings(), \
         patch.object(proxy_mod, "exponential_backoff_retry", AsyncMock(side_effect=RuntimeError("unexpected"))):
        result = await proxy_request_with_retries(
            mock_client, "v1/models",
            _make_starlette_request(), {},
        )

    assert isinstance(result, JSONResponse)
    assert result.status_code == 500


# --- RequestStreamWrapper ---


@pytest.mark.asyncio
async def test_request_stream_wrapper():
    async def fake_stream():
        yield b"chunk1"
        yield b"chunk2"

    mock_request = MagicMock()
    mock_request.stream.return_value = fake_stream()

    wrapper = RequestStreamWrapper(mock_request)
    chunks = [chunk async for chunk in wrapper]
    assert chunks == [b"chunk1", b"chunk2"]


# --- stream_multipart_post ---


def _mock_stream_client(status_code=200, body=b'{"ok": true}'):
    """Create a mock AsyncClient whose .stream() is an async context manager."""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.aread = AsyncMock(return_value=body)

    @asynccontextmanager
    async def fake_stream(**kwargs):
        yield mock_response

    mock_client = MagicMock()
    mock_client.stream = fake_stream
    return mock_client, mock_response


def _mock_stream_request():
    """Create a mock Starlette request with an async stream."""
    async def fake_body_stream():
        yield b"file-data"

    mock_request = MagicMock()
    mock_request.stream.return_value = fake_body_stream()
    return mock_request


@pytest.mark.asyncio
async def test_stream_multipart_post_json_success():
    """extract_content succeeds — returns parsed JSON."""
    mock_client, _ = _mock_stream_client(200, b'{"result": "ok"}')

    with patch.object(proxy_mod, "extract_content", return_value={"result": "ok"}):
        result = await stream_multipart_post(
            _mock_stream_request(), mock_client,
            "http://upstream/upload",
            {"content-type": "multipart/form-data", "accept": "application/json"},
        )

    assert isinstance(result, JSONResponse)
    assert result.status_code == 200


@pytest.mark.asyncio
async def test_stream_multipart_post_fallback_to_body():
    """extract_content raises, but json.loads(body) succeeds."""
    mock_client, _ = _mock_stream_client(200, b'{"fallback": true}')

    with patch.object(proxy_mod, "extract_content", side_effect=ValueError("parse failed")):
        result = await stream_multipart_post(
            _mock_stream_request(), mock_client,
            "http://upstream/upload",
            {"content-type": "multipart/form-data"},
        )

    assert isinstance(result, JSONResponse)
    assert result.status_code == 200


@pytest.mark.asyncio
async def test_stream_multipart_post_invalid_json():
    """Both extract_content and json.loads fail — returns error fallback."""
    mock_client, _ = _mock_stream_client(200, b"not json at all")

    with patch.object(proxy_mod, "extract_content", side_effect=ValueError("parse failed")):
        result = await stream_multipart_post(
            _mock_stream_request(), mock_client,
            "http://upstream/upload",
            {"content-type": "multipart/form-data"},
        )

    assert isinstance(result, JSONResponse)
    assert result.status_code == 200
    import json
    body = json.loads(result.body)
    assert body == {"error": "Invalid JSON response"}


@pytest.mark.asyncio
async def test_stream_multipart_post_filters_headers():
    """Hop-by-hop headers (content-length, transfer-encoding, etc.) are stripped."""
    mock_client, _ = _mock_stream_client()
    captured_kwargs = {}

    @asynccontextmanager
    async def capturing_stream(**kwargs):
        captured_kwargs.update(kwargs)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.aread = AsyncMock(return_value=b'{}')
        yield mock_resp

    mock_client.stream = capturing_stream

    headers = {
        "content-type": "multipart/form-data",
        "content-length": "123",
        "transfer-encoding": "chunked",
        "connection": "keep-alive",
        "expect": "100-continue",
        "host": "localhost",
        "x-custom": "keep-me",
    }

    with patch.object(proxy_mod, "extract_content", return_value={}):
        await stream_multipart_post(
            _mock_stream_request(), mock_client,
            "http://upstream/upload", headers,
        )

    forwarded = captured_kwargs["headers"]
    assert "x-custom" in forwarded
    for excluded in ("content-length", "transfer-encoding", "connection", "expect", "host"):
        assert excluded not in forwarded


# --- proxy_request_with_retries: multipart branch ---


@pytest.mark.asyncio
async def test_proxy_request_multipart_delegates_to_stream():
    """POST with multipart/form-data delegates to stream_multipart_post."""
    expected = JSONResponse(content={"uploaded": True}, status_code=200)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/llm/v1/upload",
        "query_string": b"",
        "headers": [
            (b"content-type", b"multipart/form-data; boundary=abc"),
            (b"accept", b"application/json"),
        ],
        "root_path": "",
    }
    request = Request(scope, receive=AsyncMock(return_value={"type": "http.request", "body": b""}))

    mock_client = AsyncMock(spec=AsyncClient)

    with _patch_settings(), \
         patch.object(proxy_mod, "stream_multipart_post", AsyncMock(return_value=expected)) as mock_stream:
        result = await proxy_request_with_retries(
            mock_client, "v1/upload", request, {},
        )

    assert result is expected
    mock_stream.assert_awaited_once()
