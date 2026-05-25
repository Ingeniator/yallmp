import json
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.search_hub import SearchHub, SearchProvider
from app.schemas.search import SearchProviderConfig, SearchRequest, SearchResponse, SearchResult
from app.schemas.provider import AuthConfig, AuthType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_provider(tmp_path, data: dict, filename: str | None = None) -> None:
    name = filename or f"{data['name']}.json"
    (tmp_path / name).write_text(json.dumps(data))


@pytest.fixture
def provider_dir(tmp_path):
    _write_provider(tmp_path, {
        "name": "tavily",
        "type": "tavily",
        "base_url": "https://api.tavily.com",
        "auth": {"type": "APIKEY", "api_key": "tv-key"},
        "default": True,
        "cost_per_search": 0.001,
    })
    _write_provider(tmp_path, {
        "name": "exa",
        "type": "exa",
        "base_url": "https://api.exa.ai",
        "auth": {"type": "APIKEY", "api_key": "exa-key"},
        "default": False,
    })
    return str(tmp_path)


@pytest.fixture
def hub(provider_dir):
    h = SearchHub()
    h.load_providers(provider_dir)
    return h


# ---------------------------------------------------------------------------
# load_providers
# ---------------------------------------------------------------------------

def test_load_providers(hub):
    assert "tavily" in hub.providers
    assert "exa" in hub.providers
    assert len(hub.providers) == 2


def test_load_providers_sets_default(hub):
    assert hub._default == "tavily"


def test_load_providers_missing_directory():
    h = SearchHub()
    h.load_providers("/nonexistent/path")
    assert len(h.providers) == 0


def test_load_providers_invalid_json(tmp_path):
    (tmp_path / "bad.json").write_text("not json at all")
    h = SearchHub()
    h.load_providers(str(tmp_path))
    assert len(h.providers) == 0


def test_load_providers_skips_files_without_name(tmp_path):
    (tmp_path / "no_name.json").write_text(json.dumps({"type": "tavily"}))
    h = SearchHub()
    h.load_providers(str(tmp_path))
    assert len(h.providers) == 0


def test_load_providers_rejects_duplicate_name(tmp_path):
    _write_provider(tmp_path, {"name": "dup", "type": "tavily", "base_url": "http://a"}, "a.json")
    _write_provider(tmp_path, {"name": "dup", "type": "exa", "base_url": "http://b"}, "b.json")
    h = SearchHub()
    h.load_providers(str(tmp_path))
    assert len(h.providers) == 1  # first wins


def test_load_providers_warns_on_multiple_defaults(tmp_path, caplog):
    _write_provider(tmp_path, {"name": "p1", "type": "tavily", "base_url": "http://a", "default": True})
    _write_provider(tmp_path, {"name": "p2", "type": "exa", "base_url": "http://b", "default": True})
    h = SearchHub()
    h.load_providers(str(tmp_path))
    assert h._default == "p1"  # first one wins


def test_load_providers_env_var_expansion(tmp_path):
    _write_provider(tmp_path, {
        "name": "envtest",
        "type": "brave",
        "base_url": "${TEST_SEARCH_URL}",
        "auth": {"type": "APIKEY", "api_key": "${TEST_SEARCH_KEY}"},
    })
    os.environ["TEST_SEARCH_URL"] = "https://api.search.brave.com"
    os.environ["TEST_SEARCH_KEY"] = "brave-secret"
    try:
        h = SearchHub()
        h.load_providers(str(tmp_path))
        assert h.providers["envtest"].config.base_url == "https://api.search.brave.com"
        assert h.providers["envtest"].config.auth.api_key == "brave-secret"
    finally:
        del os.environ["TEST_SEARCH_URL"]
        del os.environ["TEST_SEARCH_KEY"]


# ---------------------------------------------------------------------------
# resolve
# ---------------------------------------------------------------------------

def test_resolve_returns_default(hub):
    p = hub.resolve(None)
    assert p.config.name == "tavily"


def test_resolve_returns_named_provider(hub):
    p = hub.resolve("exa")
    assert p.config.name == "exa"


def test_resolve_unknown_provider_raises_400(hub):
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        hub.resolve("unknown-provider")
    assert exc_info.value.status_code == 400


def test_resolve_no_default_and_no_name_raises_400(tmp_path):
    from fastapi import HTTPException
    _write_provider(tmp_path, {"name": "only", "type": "exa", "base_url": "http://x", "default": False})
    h = SearchHub()
    h.load_providers(str(tmp_path))
    with pytest.raises(HTTPException) as exc_info:
        h.resolve(None)
    assert exc_info.value.status_code == 400


def test_resolve_empty_hub_raises_503():
    from fastapi import HTTPException
    h = SearchHub()
    with pytest.raises(HTTPException) as exc_info:
        h.resolve(None)
    assert exc_info.value.status_code == 503


# ---------------------------------------------------------------------------
# SearchProvider.get_auth_headers
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_auth_headers_tavily():
    config = SearchProviderConfig(
        name="tavily", type="tavily", base_url="http://x",
        auth=AuthConfig(type=AuthType.APIKEY, api_key="tv-key"),
    )
    p = SearchProvider(config)
    headers = await p.get_auth_headers()
    assert headers == {"X-Tvly-Api-Key": "tv-key"}


@pytest.mark.asyncio
async def test_get_auth_headers_exa():
    config = SearchProviderConfig(
        name="exa", type="exa", base_url="http://x",
        auth=AuthConfig(type=AuthType.APIKEY, api_key="exa-key"),
    )
    p = SearchProvider(config)
    headers = await p.get_auth_headers()
    assert headers == {"x-api-key": "exa-key"}


@pytest.mark.asyncio
async def test_get_auth_headers_brave():
    config = SearchProviderConfig(
        name="brave", type="brave", base_url="http://x",
        auth=AuthConfig(type=AuthType.APIKEY, api_key="brave-key"),
    )
    p = SearchProvider(config)
    headers = await p.get_auth_headers()
    assert headers == {"X-Subscription-Token": "brave-key"}


@pytest.mark.asyncio
async def test_get_auth_headers_none():
    config = SearchProviderConfig(
        name="open", type="brave", base_url="http://x",
        auth=AuthConfig(type=AuthType.NONE),
    )
    p = SearchProvider(config)
    headers = await p.get_auth_headers()
    assert headers == {}


# ---------------------------------------------------------------------------
# SearchProvider.search — circuit breaker
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_raises_503_when_circuit_open():
    from fastapi import HTTPException
    config = SearchProviderConfig(
        name="cb-test", type="tavily", base_url="http://x",
        auth=AuthConfig(type=AuthType.APIKEY, api_key="k"),
    )
    p = SearchProvider(config)
    await p.startup()

    # Force circuit open
    p.circuit_breaker.is_open = True
    import time
    p.circuit_breaker.open_time = time.time()  # just opened
    p.circuit_breaker._recovery_time = 30

    req = SearchRequest(query="test")
    with pytest.raises(HTTPException) as exc_info:
        await p.search(req)
    assert exc_info.value.status_code == 503

    await p.shutdown()


# ---------------------------------------------------------------------------
# startup / shutdown
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hub_startup_shutdown(provider_dir):
    h = SearchHub()
    h.load_providers(provider_dir)
    await h.startup()
    for p in h.providers.values():
        assert p.client is not None
    await h.shutdown()
