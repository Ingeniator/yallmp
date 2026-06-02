import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.responses import JSONResponse
from httpx import Response as HTTPXResponse, Request as HTTPXRequest
from starlette.requests import Request

from app.core.proxy import proxy_request_to_provider, _strip_model_prefix
from app.schemas.provider import LlmProviderConfig, AuthConfig, AuthType, AliasEntry
from app.services.llm_hub import LlmHub, LlmProvider


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


# --- alias fallback routing tests ---

def _mock_settings_patch():
    m = MagicMock()
    m.proxy_exclude_headers = "host"
    m.proxy_max_retries = 0
    m.proxy_base_delay = 0.01
    m.proxy_backoff_factor = 2.0
    m.proxy_failure_threshold = 0
    m.proxy_window_size = 60
    m.proxy_recovery_time = 30
    m.billing_enabled = False
    m.tracing_enabled = False
    return m


def _build_hub_with_alias(primary_response, fallback_response):
    """Build a LlmHub with two providers and one alias wiring them together."""
    hub = LlmHub()

    for prefix, mock_resp in [("primary-provider", primary_response), ("fallback-provider", fallback_response)]:
        config = LlmProviderConfig(
            prefix=prefix,
            base_url=f"http://{prefix}.api",
            auth=AuthConfig(type=AuthType.NONE),
            max_retries=0,
            base_delay=0.01,
        )
        provider = LlmProvider(config)
        provider.client = AsyncMock()
        provider.client.request = AsyncMock(return_value=mock_resp)
        provider.circuit_breaker = MagicMock()
        provider.circuit_breaker.check_open = AsyncMock(return_value=False)
        provider.circuit_breaker.record_success = AsyncMock()
        provider.circuit_breaker.record_failure = AsyncMock(return_value=False)
        hub.providers[prefix] = provider

    hub.aliases["smart"] = AliasEntry(
        target="primary-provider/fast-model",
        fallback="fallback-provider/safe-model",
    )
    return hub


@pytest.mark.asyncio
async def test_alias_uses_primary_on_success():
    ok_resp = HTTPXResponse(
        status_code=200,
        json={"choices": [], "usage": {}},
        request=HTTPXRequest("POST", "http://primary-provider.api/v1/chat/completions"),
    )
    hub = _build_hub_with_alias(primary_response=ok_resp, fallback_response=ok_resp)

    alias = hub.resolve_alias("smart")
    primary = hub.resolve_model(alias.target)
    assert primary is not None
    p_provider, p_stripped = primary

    body = json.dumps({"model": "smart", "messages": []}).encode()
    request = _make_request(body=body)

    with patch("app.core.proxy.settings", _mock_settings_patch()):
        response = await proxy_request_to_provider(
            provider=p_provider,
            path="v1/chat/completions",
            request=request,
            auth_headers={},
            original_model="smart",
            stripped_model=p_stripped,
        )

    assert response.status_code == 200
    # fallback provider should not have been called
    hub.providers["fallback-provider"].client.request.assert_not_called()


@pytest.mark.asyncio
async def test_alias_falls_back_when_primary_fails():
    """Primary returns 500 → fallback is tried and returns 200."""
    fail_resp = HTTPXResponse(
        status_code=500,
        json={"error": "upstream error"},
        request=HTTPXRequest("POST", "http://primary-provider.api/v1/chat/completions"),
    )
    ok_resp = HTTPXResponse(
        status_code=200,
        json={"choices": [{"message": {"content": "ok"}}], "usage": {}},
        request=HTTPXRequest("POST", "http://fallback-provider.api/v1/chat/completions"),
    )
    hub = _build_hub_with_alias(primary_response=fail_resp, fallback_response=ok_resp)

    alias = hub.resolve_alias("smart")
    primary = hub.resolve_model(alias.target)
    p_provider, p_stripped = primary

    body = json.dumps({"model": "smart", "messages": []}).encode()
    request = _make_request(body=body)

    settings_mock = _mock_settings_patch()
    with patch("app.core.proxy.settings", settings_mock):
        primary_response = await proxy_request_to_provider(
            provider=p_provider,
            path="v1/chat/completions",
            request=request,
            auth_headers={},
            original_model="smart",
            stripped_model=p_stripped,
        )

        # Simulate app.py fallback logic
        assert alias.fallback is not None
        assert isinstance(primary_response, JSONResponse)
        assert primary_response.status_code >= 400

        fb = hub.resolve_model(alias.fallback)
        assert fb is not None
        fb_provider, fb_stripped = fb

        final_response = await proxy_request_to_provider(
            provider=fb_provider,
            path="v1/chat/completions",
            request=request,
            auth_headers={},
            original_model="smart",
            stripped_model=fb_stripped,
        )

    assert final_response.status_code == 200
    assert fb_stripped == "safe-model"


@pytest.mark.asyncio
async def test_alias_no_fallback_returns_primary_error():
    """Alias without fallback returns the primary error as-is."""
    fail_resp = HTTPXResponse(
        status_code=503,
        json={"error": "service unavailable"},
        request=HTTPXRequest("POST", "http://primary-provider.api/v1/chat/completions"),
    )
    hub = _build_hub_with_alias(primary_response=fail_resp, fallback_response=fail_resp)
    hub.aliases["no-fallback"] = AliasEntry(target="primary-provider/fast-model")

    alias = hub.resolve_alias("no-fallback")
    assert alias.fallback is None

    primary = hub.resolve_model(alias.target)
    p_provider, p_stripped = primary

    body = json.dumps({"model": "no-fallback", "messages": []}).encode()
    request = _make_request(body=body)

    with patch("app.core.proxy.settings", _mock_settings_patch()):
        response = await proxy_request_to_provider(
            provider=p_provider,
            path="v1/chat/completions",
            request=request,
            auth_headers={},
            original_model="no-fallback",
            stripped_model=p_stripped,
        )

    assert response.status_code == 503


# --- single-provider alias rewrite tests ---

def _mock_proxy_settings():
    s = MagicMock()
    s.proxy_target_url = "http://upstream.api"
    s.proxy_failure_threshold = 3
    s.proxy_recovery_time = 30
    s.proxy_window_size = 10
    s.proxy_exclude_headers = ""
    s.tracing_enabled = False
    return s


@pytest.mark.asyncio
async def test_proxy_request_with_retries_alias_rewrite():
    """proxy_request_with_retries rewrites model name when body override is supplied."""
    from app.core.proxy import proxy_request_with_retries

    original_body = json.dumps({"model": "smart", "messages": []}).encode()
    rewritten_body = json.dumps({"model": "gpt-4o", "messages": []}).encode()
    request = _make_request(body=original_body)

    captured_body = {}

    async def fake_do_proxy(client, target_url, method, headers, body, path, request, pricing_cache=None, **kw):
        captured_body["value"] = body
        return JSONResponse(content={"ok": True}, status_code=200)

    mock_client = AsyncMock()
    with (
        patch("app.core.proxy.settings", _mock_proxy_settings()),
        patch("app.core.proxy._do_proxy_request", side_effect=fake_do_proxy),
    ):
        await proxy_request_with_retries(mock_client, "v1/chat/completions", request, body=rewritten_body)

    assert captured_body["value"] == rewritten_body
    assert json.loads(captured_body["value"])["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_proxy_request_with_retries_no_override_uses_request_body():
    """proxy_request_with_retries reads body from request when no override given."""
    from app.core.proxy import proxy_request_with_retries

    original_body = json.dumps({"model": "gpt-4o", "messages": []}).encode()
    request = _make_request(body=original_body)

    captured_body = {}

    async def fake_do_proxy(client, target_url, method, headers, body, path, request, pricing_cache=None, **kw):
        captured_body["value"] = body
        return JSONResponse(content={"ok": True}, status_code=200)

    mock_client = AsyncMock()
    with (
        patch("app.core.proxy.settings", _mock_proxy_settings()),
        patch("app.core.proxy._do_proxy_request", side_effect=fake_do_proxy),
    ):
        await proxy_request_with_retries(mock_client, "v1/chat/completions", request)

    assert captured_body["value"] == original_body
