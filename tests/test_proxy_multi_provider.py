import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.responses import JSONResponse
from httpx import Response as HTTPXResponse, Request as HTTPXRequest
from starlette.requests import Request

from app.core.proxy import proxy_request_to_provider, _strip_model_prefix
from app.schemas.provider import LlmProviderConfig, AuthConfig, AuthType


def _make_request(method="POST", path="/llm/v1/chat/completions", body=b"", headers=None):
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


class FakeProvider:
    """Minimal provider stand-in for tests."""
    def __init__(self, prefix="test", base_url="http://test.api/v1"):
        self.config = LlmProviderConfig(
            prefix=prefix,
            base_url=base_url,
            auth=AuthConfig(type=AuthType.NONE),
            max_retries=0,
            base_delay=0.01,
        )
        self.client = AsyncMock()
        self.circuit_breaker = MagicMock()
        self.circuit_breaker.check_open = AsyncMock(return_value=False)
        self.circuit_breaker.record_success = AsyncMock()
        self.circuit_breaker.record_failure = AsyncMock(return_value=False)


# --- _strip_model_prefix tests ---

def test_strip_model_prefix_replaces_model():
    body = json.dumps({"model": "gigachat/DeepSeek-R1", "messages": []}).encode()
    result = _strip_model_prefix(body, "gigachat/DeepSeek-R1", "DeepSeek-R1")
    parsed = json.loads(result)
    assert parsed["model"] == "DeepSeek-R1"
    assert parsed["messages"] == []


def test_strip_model_prefix_no_match():
    body = json.dumps({"model": "other-model"}).encode()
    result = _strip_model_prefix(body, "gigachat/DeepSeek-R1", "DeepSeek-R1")
    parsed = json.loads(result)
    assert parsed["model"] == "other-model"


def test_strip_model_prefix_invalid_json():
    body = b"not json"
    result = _strip_model_prefix(body, "a/b", "b")
    assert result == b"not json"


def test_strip_model_prefix_no_model_field():
    body = json.dumps({"messages": []}).encode()
    result = _strip_model_prefix(body, "a/b", "b")
    parsed = json.loads(result)
    assert "model" not in parsed or parsed.get("model") is None


# --- proxy_request_to_provider tests ---

@pytest.mark.asyncio
async def test_provider_routing_success():
    provider = FakeProvider()
    mock_response = HTTPXResponse(
        status_code=200,
        json={"choices": [], "usage": {}},
        request=HTTPXRequest("POST", "http://test.api/v1/v1/chat/completions"),
    )
    provider.client.request = AsyncMock(return_value=mock_response)

    body = json.dumps({"model": "test/mymodel", "messages": []}).encode()
    request = _make_request(body=body)

    with patch("app.core.proxy.settings") as mock_settings:
        mock_settings.proxy_exclude_headers = "host"
        mock_settings.proxy_max_retries = 0
        mock_settings.proxy_base_delay = 0.01
        mock_settings.proxy_backoff_factor = 2.0
        mock_settings.proxy_failure_threshold = 0
        mock_settings.proxy_window_size = 60
        mock_settings.proxy_recovery_time = 30

        response = await proxy_request_to_provider(
            provider=provider,
            path="v1/chat/completions",
            request=request,
            auth_headers={"Authorization": "Bearer tok"},
            original_model="test/mymodel",
            stripped_model="mymodel",
        )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_provider_routing_strips_model_in_body():
    provider = FakeProvider()
    captured_content = {}

    async def capture_request(method, url, headers=None, content=None):
        captured_content["body"] = content
        return HTTPXResponse(
            status_code=200,
            json={"choices": []},
            request=HTTPXRequest("POST", url),
        )

    provider.client.request = capture_request

    body = json.dumps({"model": "test/mymodel", "messages": [], "stream": False}).encode()
    request = _make_request(body=body)

    with patch("app.core.proxy.settings") as mock_settings:
        mock_settings.proxy_exclude_headers = "host"
        mock_settings.proxy_max_retries = 0
        mock_settings.proxy_base_delay = 0.01
        mock_settings.proxy_backoff_factor = 2.0
        mock_settings.proxy_failure_threshold = 0
        mock_settings.proxy_window_size = 60
        mock_settings.proxy_recovery_time = 30

        await proxy_request_to_provider(
            provider=provider,
            path="v1/chat/completions",
            request=request,
            auth_headers={},
            original_model="test/mymodel",
            stripped_model="mymodel",
        )

    sent_body = json.loads(captured_content["body"])
    assert sent_body["model"] == "mymodel"


@pytest.mark.asyncio
async def test_provider_routing_uses_provider_base_url():
    provider = FakeProvider(base_url="http://custom.api")
    captured_url = {}

    async def capture_request(method, url, headers=None, content=None):
        captured_url["url"] = url
        return HTTPXResponse(
            status_code=200,
            json={"choices": []},
            request=HTTPXRequest("POST", url),
        )

    provider.client.request = capture_request

    body = json.dumps({"model": "test/m", "messages": [], "stream": False}).encode()
    request = _make_request(body=body)

    with patch("app.core.proxy.settings") as mock_settings:
        mock_settings.proxy_exclude_headers = "host"
        mock_settings.proxy_max_retries = 0
        mock_settings.proxy_base_delay = 0.01
        mock_settings.proxy_backoff_factor = 2.0
        mock_settings.proxy_failure_threshold = 0
        mock_settings.proxy_window_size = 60
        mock_settings.proxy_recovery_time = 30

        await proxy_request_to_provider(
            provider=provider,
            path="v1/chat/completions",
            request=request,
            auth_headers={},
            original_model="test/m",
            stripped_model="m",
        )

    assert captured_url["url"].startswith("http://custom.api/")


@pytest.mark.asyncio
async def test_provider_routing_error_response():
    provider = FakeProvider()
    mock_response = HTTPXResponse(
        status_code=500,
        json={"error": "internal"},
        request=HTTPXRequest("POST", "http://test.api/v1/v1/chat/completions"),
    )
    provider.client.request = AsyncMock(return_value=mock_response)

    body = json.dumps({"model": "test/m", "messages": []}).encode()
    request = _make_request(body=body)

    with patch("app.core.proxy.settings") as mock_settings:
        mock_settings.proxy_exclude_headers = "host"
        mock_settings.proxy_max_retries = 0
        mock_settings.proxy_base_delay = 0.01
        mock_settings.proxy_backoff_factor = 2.0
        mock_settings.proxy_failure_threshold = 0
        mock_settings.proxy_window_size = 60
        mock_settings.proxy_recovery_time = 30

        response = await proxy_request_to_provider(
            provider=provider,
            path="v1/chat/completions",
            request=request,
            auth_headers={},
            original_model="test/m",
            stripped_model="m",
        )

    assert response.status_code == 500
