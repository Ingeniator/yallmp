import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import AsyncClient, Response as HTTPXResponse, Request as HTTPXRequest
from starlette.requests import Request
from starlette.responses import StreamingResponse


@pytest.mark.asyncio
async def test_proxy_passes_query_parameters():
    """Query parameters from the original request must be forwarded to the upstream."""
    from app.core.proxy import proxy_request_with_retries

    # Build a fake Starlette request with query params
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/llm/v1/models",
        "query_string": b"limit=10&offset=0",
        "headers": [
            (b"host", b"localhost"),
            (b"accept", b"application/json"),
        ],
        "root_path": "",
    }
    request = Request(scope, receive=AsyncMock(return_value={"type": "http.request", "body": b""}))

    # Mock the httpx AsyncClient
    mock_response = HTTPXResponse(
        status_code=200,
        json={"data": []},
        request=HTTPXRequest("GET", "http://upstream/v1/models?limit=10&offset=0"),
    )
    mock_client = AsyncMock(spec=AsyncClient)
    mock_client.request = AsyncMock(return_value=mock_response)

    with patch("app.core.proxy.settings") as mock_settings:
        mock_settings.proxy_target_url = "http://upstream"
        mock_settings.proxy_exclude_headers = "host,authorization"
        mock_settings.proxy_max_retries = 0
        mock_settings.proxy_base_delay = 0.1
        mock_settings.proxy_backoff_factor = 2.0
        mock_settings.proxy_failure_threshold = 0
        mock_settings.proxy_window_size = 60
        mock_settings.proxy_recovery_time = 30

        await proxy_request_with_retries(
            client=mock_client,
            path="v1/models",
            request=request,
            custom_headers={},
        )

    # Verify the upstream URL included query parameters
    call_args = mock_client.request.call_args
    called_url = call_args[0][1]  # positional: method, url
    assert "?" in called_url, f"Query parameters missing from upstream URL: {called_url}"
    assert "limit=10" in called_url
    assert "offset=0" in called_url


@pytest.mark.asyncio
async def test_proxy_works_without_query_parameters():
    """Requests without query params should not append a '?' to the URL."""
    from app.core.proxy import proxy_request_with_retries

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/llm/v1/models",
        "query_string": b"",
        "headers": [
            (b"host", b"localhost"),
            (b"accept", b"application/json"),
        ],
        "root_path": "",
    }
    request = Request(scope, receive=AsyncMock(return_value={"type": "http.request", "body": b""}))

    mock_response = HTTPXResponse(
        status_code=200,
        json={"data": []},
        request=HTTPXRequest("GET", "http://upstream/v1/models"),
    )
    mock_client = AsyncMock(spec=AsyncClient)
    mock_client.request = AsyncMock(return_value=mock_response)

    with patch("app.core.proxy.settings") as mock_settings:
        mock_settings.proxy_target_url = "http://upstream"
        mock_settings.proxy_exclude_headers = "host,authorization"
        mock_settings.proxy_max_retries = 0
        mock_settings.proxy_base_delay = 0.1
        mock_settings.proxy_backoff_factor = 2.0
        mock_settings.proxy_failure_threshold = 0
        mock_settings.proxy_window_size = 60
        mock_settings.proxy_recovery_time = 30

        await proxy_request_with_retries(
            client=mock_client,
            path="v1/models",
            request=request,
            custom_headers={},
        )

    called_url = mock_client.request.call_args[0][1]
    assert called_url == "http://upstream/v1/models", f"Unexpected URL: {called_url}"
    assert "?" not in called_url


# --- Helpers for streaming tests ---

def _mock_settings():
    """Return a patch context manager for proxy settings."""
    return patch("app.core.proxy.settings")


def _make_request(method="POST", path="/llm/v1/chat/completions", body=b"", headers=None):
    """Create a fake Starlette Request."""
    raw_headers = [
        (b"host", b"localhost"),
        (b"content-type", b"application/json"),
    ]
    if headers:
        for k, v in headers.items():
            raw_headers.append((k.encode(), v.encode()))
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "query_string": b"",
        "headers": raw_headers,
        "root_path": "",
    }
    return Request(scope, receive=AsyncMock(return_value={"type": "http.request", "body": body}))


def _configure_settings(mock_settings):
    mock_settings.proxy_target_url = "http://upstream"
    mock_settings.proxy_exclude_headers = "host"
    mock_settings.proxy_max_retries = 0
    mock_settings.proxy_base_delay = 0.1
    mock_settings.proxy_backoff_factor = 2.0
    mock_settings.proxy_failure_threshold = 0
    mock_settings.proxy_window_size = 60
    mock_settings.proxy_recovery_time = 30


class FakeUpstreamResponse:
    """Mimics an httpx streaming response for testing."""

    def __init__(self, status_code=200, chunks=None, body=None):
        self.status_code = status_code
        self._chunks = chunks or []
        self._body = body

    async def aiter_bytes(self):
        for chunk in self._chunks:
            if isinstance(chunk, str):
                yield chunk.encode()
            else:
                yield chunk

    async def aread(self):
        return self._body or b""

    async def aclose(self):
        pass


@pytest.mark.asyncio
async def test_streaming_request_returns_streaming_response():
    """POST with stream=true should return a StreamingResponse with SSE media type."""
    from app.core.proxy import proxy_request_with_retries

    body = json.dumps({"model": "test", "messages": [], "stream": True}).encode()
    request = _make_request(body=body)

    sse_chunks = [
        "data: {\"id\":\"1\",\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n",
        "data: {\"id\":\"1\",\"choices\":[{\"delta\":{\"content\":\" world\"}}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7},\"model\":\"test-model\"}\n\n",
        "data: [DONE]\n\n",
    ]
    fake_upstream = FakeUpstreamResponse(status_code=200, chunks=sse_chunks)

    mock_client = AsyncMock(spec=AsyncClient)
    mock_client.build_request = MagicMock(return_value=MagicMock())
    mock_client.send = AsyncMock(return_value=fake_upstream)

    with _mock_settings() as mock_settings:
        _configure_settings(mock_settings)
        response = await proxy_request_with_retries(
            client=mock_client, path="v1/chat/completions",
            request=request, custom_headers={},
        )

    assert isinstance(response, StreamingResponse)
    assert response.media_type == "text/event-stream"

    # Consume the stream and verify chunks are forwarded
    collected = b""
    async for chunk in response.body_iterator:
        collected += chunk if isinstance(chunk, bytes) else chunk.encode()

    assert b"Hello" in collected
    assert b"world" in collected
    assert b"[DONE]" in collected


@pytest.mark.asyncio
async def test_streaming_request_upstream_error_returns_json():
    """If upstream returns non-2xx for a streaming request, return JSONResponse."""
    from app.core.proxy import proxy_request_with_retries

    body = json.dumps({"model": "test", "messages": [], "stream": True}).encode()
    request = _make_request(body=body)

    error_body = json.dumps({"error": "model not found"}).encode()
    fake_upstream = FakeUpstreamResponse(status_code=404, body=error_body)

    mock_client = AsyncMock(spec=AsyncClient)
    mock_client.build_request = MagicMock(return_value=MagicMock())
    mock_client.send = AsyncMock(return_value=fake_upstream)

    with _mock_settings() as mock_settings:
        _configure_settings(mock_settings)
        response = await proxy_request_with_retries(
            client=mock_client, path="v1/chat/completions",
            request=request, custom_headers={},
        )

    from fastapi.responses import JSONResponse
    assert isinstance(response, JSONResponse)
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_non_streaming_post_returns_json_response():
    """POST without stream=true should still return a regular JSONResponse."""
    from app.core.proxy import proxy_request_with_retries

    body = json.dumps({"model": "test", "messages": [], "stream": False}).encode()
    request = _make_request(body=body)

    mock_response = HTTPXResponse(
        status_code=200,
        json={"id": "1", "choices": [], "usage": {}},
        request=HTTPXRequest("POST", "http://upstream/v1/chat/completions"),
    )
    mock_client = AsyncMock(spec=AsyncClient)
    mock_client.request = AsyncMock(return_value=mock_response)

    with _mock_settings() as mock_settings:
        _configure_settings(mock_settings)
        response = await proxy_request_with_retries(
            client=mock_client, path="v1/chat/completions",
            request=request, custom_headers={},
        )

    from fastapi.responses import JSONResponse
    assert isinstance(response, JSONResponse)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_streaming_connection_failure_returns_502():
    """Connection failure on streaming request should return 502 JSONResponse."""
    from app.core.proxy import proxy_request_with_retries
    from httpx import ConnectError

    body = json.dumps({"model": "test", "messages": [], "stream": True}).encode()
    request = _make_request(body=body)

    mock_client = AsyncMock(spec=AsyncClient)
    mock_client.build_request = MagicMock(return_value=MagicMock())
    mock_client.send = AsyncMock(side_effect=ConnectError("connection refused"))

    with _mock_settings() as mock_settings:
        _configure_settings(mock_settings)
        response = await proxy_request_with_retries(
            client=mock_client, path="v1/chat/completions",
            request=request, custom_headers={},
        )

    from fastapi.responses import JSONResponse
    assert isinstance(response, JSONResponse)


# ---------------------------------------------------------------------------
# _assemble_streaming_output
# ---------------------------------------------------------------------------

class TestAssembleStreamingOutput:
    """Unit tests for _assemble_streaming_output."""

    def _full_text(self, payloads: list[dict]) -> str:
        lines = [f"data: {json.dumps(p)}\n\n" for p in payloads]
        lines.append("data: [DONE]\n\n")
        return "".join(lines)

    def test_concatenates_delta_content(self):
        from app.core.proxy import _assemble_streaming_output

        chunks = [
            {"id": "x", "model": "m", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello"}, "finish_reason": None}]},
            {"id": "x", "model": "m", "choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": None}]},
            {"id": "x", "model": "m", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}},
        ]
        last = chunks[-1]
        result = _assemble_streaming_output(self._full_text(chunks), last)

        assert result["choices"][0]["message"]["content"] == "Hello world"
        assert result["choices"][0]["message"]["role"] == "assistant"
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_model_id_usage_from_last_payload(self):
        from app.core.proxy import _assemble_streaming_output

        chunks = [
            {"id": "abc", "model": "gpt-x", "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}]},
            {"id": "abc", "model": "gpt-x", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}},
        ]
        last = chunks[-1]
        result = _assemble_streaming_output(self._full_text(chunks), last)

        assert result["id"] == "abc"
        assert result["model"] == "gpt-x"
        assert result["object"] == "chat.completion"
        assert result["usage"]["total_tokens"] == 4

    def test_multiple_choice_indices(self):
        from app.core.proxy import _assemble_streaming_output

        chunks = [
            {"id": "y", "model": "m", "choices": [
                {"index": 0, "delta": {"content": "A"}, "finish_reason": None},
                {"index": 1, "delta": {"content": "B"}, "finish_reason": None},
            ]},
            {"id": "y", "model": "m", "choices": [
                {"index": 0, "delta": {"content": "1"}, "finish_reason": "stop"},
                {"index": 1, "delta": {"content": "2"}, "finish_reason": "stop"},
            ], "usage": {}},
        ]
        last = chunks[-1]
        result = _assemble_streaming_output(self._full_text(chunks), last)

        assert len(result["choices"]) == 2
        assert result["choices"][0]["message"]["content"] == "A1"
        assert result["choices"][1]["message"]["content"] == "B2"

    def test_skips_done_and_malformed_lines(self):
        from app.core.proxy import _assemble_streaming_output

        full_text = (
            "data: [DONE]\n\n"
            "data: not-json\n\n"
            'data: {"id":"z","model":"m","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}],"usage":{}}\n\n'
        )
        last = {"id": "z", "model": "m", "usage": {}}
        result = _assemble_streaming_output(full_text, last)

        assert result["choices"][0]["message"]["content"] == "ok"

    def test_empty_stream_returns_empty_choices(self):
        from app.core.proxy import _assemble_streaming_output

        result = _assemble_streaming_output("data: [DONE]\n\n", {"id": "", "model": "", "usage": None})

        assert result["choices"] == []
        assert result["object"] == "chat.completion"


# ---------------------------------------------------------------------------
# _emit_streaming_metrics — output_body uses assembled content
# ---------------------------------------------------------------------------

class TestEmitStreamingMetricsOutputBody:
    """Verify _emit_streaming_metrics passes assembled output to trace_proxy_request."""

    def _make_request(self):
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "query_string": b"",
            "headers": [
                (b"host", b"localhost"),
                (b"x-group-id", b"g1"),
            ],
            "root_path": "",
        }
        return Request(scope, receive=AsyncMock(return_value={"type": "http.request", "body": b""}))

    def test_output_body_is_assembled_not_last_chunk(self):
        from app.core.proxy import _emit_streaming_metrics

        sse_chunks = [
            'data: {"id":"1","model":"gpt-x","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}\n\n',
            'data: {"id":"1","model":"gpt-x","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}\n\n',
            'data: {"id":"1","model":"gpt-x","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}\n\n',
            "data: [DONE]\n\n",
        ]
        request = self._make_request()

        with patch("app.core.proxy.trace_proxy_request") as mock_trace, \
             patch("app.core.proxy.settings") as mock_settings:
            mock_settings.tracing_log_io = True
            mock_settings.billing_enabled = False
            _emit_streaming_metrics(sse_chunks, request, start_time=0)

        mock_trace.assert_called_once()
        output_body = mock_trace.call_args.kwargs["output_body"]
        assert output_body["object"] == "chat.completion"
        assert output_body["choices"][0]["message"]["content"] == "Hello world"
        assert output_body["choices"][0]["finish_reason"] == "stop"
        assert output_body["usage"]["total_tokens"] == 7
