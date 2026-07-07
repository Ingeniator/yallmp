import pytest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import Response as HTTPXResponse, Request as HTTPXRequest, AsyncClient
from fastapi.responses import JSONResponse
from starlette.requests import Request

import app.core.proxy as proxy_mod
from app.core.proxy import (
    _parse_model_version,
    _extract_tools_defined,
    _extract_tool_calls,
    _extract_streaming_tool_calls,
    _extract_output_tool_calls,
    _tool_calls_from_response,
    _normalize_usage,
    _is_traceable_path,
    _unwrap_responses_event,
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


# --- _extract_tools_defined ---


def test_extract_tools_defined_none():
    assert _extract_tools_defined(None) == []


def test_extract_tools_defined_no_tools_key():
    assert _extract_tools_defined({"messages": []}) == []


def test_extract_tools_defined_empty_list():
    assert _extract_tools_defined({"tools": []}) == []


def test_extract_tools_defined_openai_format():
    input_body = {
        "tools": [
            {"type": "function", "function": {"name": "get_weather", "description": "..."}},
            {"type": "function", "function": {"name": "search_web"}},
        ]
    }
    assert _extract_tools_defined(input_body) == ["get_weather", "search_web"]


def test_extract_tools_defined_skips_malformed():
    input_body = {
        "tools": [
            {"type": "function", "function": {"name": "valid"}},
            "not-a-dict",
            {"type": "function", "function": {}},  # missing name
            {"type": "function"},  # missing function key
        ]
    }
    assert _extract_tools_defined(input_body) == ["valid"]


def test_extract_tools_defined_responses_api_flat_format():
    """Responses API tools put "name" at the top level, not nested under "function"."""
    input_body = {
        "tools": [
            {"type": "function", "name": "get_weather", "parameters": {}},
            {"type": "function", "name": "search_web"},
        ]
    }
    assert _extract_tools_defined(input_body) == ["get_weather", "search_web"]


# --- _extract_output_tool_calls / _tool_calls_from_response ---


def test_extract_output_tool_calls_empty():
    assert _extract_output_tool_calls([]) == []


def test_extract_output_tool_calls_single():
    output = [
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hi"}]},
        {"type": "function_call", "name": "get_weather", "arguments": "{}", "call_id": "call_1"},
    ]
    assert _extract_output_tool_calls(output) == ["get_weather"]


def test_extract_output_tool_calls_skips_non_function_call_items():
    output = [{"type": "message", "role": "assistant"}, "not-a-dict"]
    assert _extract_output_tool_calls(output) == []


def test_tool_calls_from_response_dispatches_chat_completions():
    response_data = {"choices": [{"message": {"tool_calls": [{"function": {"name": "fn_a"}}]}}]}
    assert _tool_calls_from_response(response_data) == ["fn_a"]


def test_tool_calls_from_response_dispatches_responses_api():
    response_data = {"output": [{"type": "function_call", "name": "fn_b"}]}
    assert _tool_calls_from_response(response_data) == ["fn_b"]


# --- _normalize_usage ---


def test_normalize_usage_none():
    assert _normalize_usage(None) is None


def test_normalize_usage_passes_through_chat_completions_shape():
    usage = {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
    assert _normalize_usage(usage) == usage


def test_normalize_usage_aliases_responses_api_shape():
    usage = {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}
    normalized = _normalize_usage(usage)
    assert normalized["prompt_tokens"] == 10
    assert normalized["completion_tokens"] == 4
    assert normalized["input_tokens"] == 10  # original keys preserved
    assert normalized["output_tokens"] == 4


def test_normalize_usage_aliases_reasoning_tokens():
    usage = {
        "input_tokens": 10,
        "output_tokens": 4,
        "output_tokens_details": {"reasoning_tokens": 2},
    }
    normalized = _normalize_usage(usage)
    assert normalized["completion_tokens_details"]["reasoning_tokens"] == 2


# --- _is_traceable_path ---


def test_is_traceable_path_chat_completions():
    assert _is_traceable_path("v1/chat/completions") is True


def test_is_traceable_path_responses():
    assert _is_traceable_path("v1/responses") is True


def test_is_traceable_path_embeddings_untouched():
    assert _is_traceable_path("v1/embeddings") is False


# --- _unwrap_responses_event ---


def test_unwrap_responses_event_completed():
    payload = {"type": "response.completed", "response": {"id": "resp_1", "model": "gpt-x", "usage": {}}}
    unwrapped, is_responses_api = _unwrap_responses_event(payload)
    assert is_responses_api is True
    assert unwrapped == {"id": "resp_1", "model": "gpt-x", "usage": {}}


def test_unwrap_responses_event_chat_completions_passthrough():
    payload = {"id": "1", "choices": [{"delta": {"content": "hi"}}]}
    unwrapped, is_responses_api = _unwrap_responses_event(payload)
    assert is_responses_api is False
    assert unwrapped is payload


def test_unwrap_responses_event_missing_response_key():
    payload = {"type": "response.output_text.delta", "delta": "hi"}
    unwrapped, is_responses_api = _unwrap_responses_event(payload)
    assert is_responses_api is False
    assert unwrapped is payload


# --- _extract_tool_calls ---


def test_extract_tool_calls_empty():
    assert _extract_tool_calls([]) == []


def test_extract_tool_calls_no_tool_calls():
    choices = [{"message": {"role": "assistant", "content": "hello"}}]
    assert _extract_tool_calls(choices) == []


def test_extract_tool_calls_single():
    choices = [
        {
            "message": {
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
                ]
            }
        }
    ]
    assert _extract_tool_calls(choices) == ["get_weather"]


def test_extract_tool_calls_multiple():
    choices = [
        {
            "message": {
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "fn_a", "arguments": "{}"}},
                    {"id": "call_2", "function": {"name": "fn_b", "arguments": "{}"}},
                ]
            }
        }
    ]
    assert _extract_tool_calls(choices) == ["fn_a", "fn_b"]


def test_extract_tool_calls_null_tool_calls():
    choices = [{"message": {"tool_calls": None}}]
    assert _extract_tool_calls(choices) == []


# --- _extract_streaming_tool_calls ---


def test_extract_streaming_tool_calls_no_tool_calls():
    full_text = (
        'data: {"choices": [{"delta": {"content": "hello"}}]}\n'
        "data: [DONE]\n"
    )
    assert _extract_streaming_tool_calls(full_text) == []


def test_extract_streaming_tool_calls_single():
    full_text = (
        'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "get_weather", "arguments": ""}}]}}]}\n'
        'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\\"loc\\": \\"NYC\\"}"}}]}}]}\n'
        "data: [DONE]\n"
    )
    assert _extract_streaming_tool_calls(full_text) == ["get_weather"]


def test_extract_streaming_tool_calls_deduplicates():
    # Same name appearing in multiple chunks (argument fragments) — should appear once
    full_text = (
        'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "fn_x", "arguments": ""}}]}}]}\n'
        'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "fn_x", "arguments": "end"}}]}}]}\n'
        "data: [DONE]\n"
    )
    assert _extract_streaming_tool_calls(full_text) == ["fn_x"]


def test_extract_streaming_tool_calls_multiple_distinct():
    full_text = (
        'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "fn_a", "arguments": ""}}]}}]}\n'
        'data: {"choices": [{"delta": {"tool_calls": [{"index": 1, "function": {"name": "fn_b", "arguments": ""}}]}}]}\n'
        "data: [DONE]\n"
    )
    result = _extract_streaming_tool_calls(full_text)
    assert result == ["fn_a", "fn_b"]


def test_extract_streaming_tool_calls_skips_malformed_lines():
    full_text = (
        "data: not-valid-json\n"
        'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "ok"}}]}}]}\n'
        "data: [DONE]\n"
    )
    assert _extract_streaming_tool_calls(full_text) == ["ok"]


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
