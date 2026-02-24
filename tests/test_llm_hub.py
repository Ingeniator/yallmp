import pytest
import json
import os
import tempfile
from unittest.mock import AsyncMock, patch, MagicMock

from app.services.llm_hub import LlmHub, LlmProvider
from app.schemas.provider import LlmProviderConfig, AuthType, AuthConfig


@pytest.fixture
def provider_dir(tmp_path):
    """Create a temp directory with provider JSON files."""
    p1 = {
        "prefix": "alpha",
        "base_url": "http://alpha.api/v1",
        "auth": {"type": "APIKEY", "api_key": "key-alpha"},
        "models": ["model-a", "model-b"],
    }
    p2 = {
        "prefix": "beta",
        "base_url": "http://beta.api/v1",
        "auth": {"type": "NONE"},
        "models": ["model-x"],
    }
    (tmp_path / "alpha.json").write_text(json.dumps(p1))
    (tmp_path / "beta.json").write_text(json.dumps(p2))
    return str(tmp_path)


@pytest.fixture
def hub(provider_dir):
    h = LlmHub()
    h.load_providers(provider_dir)
    return h


def test_load_providers(hub):
    assert "alpha" in hub.providers
    assert "beta" in hub.providers
    assert len(hub.providers) == 2


def test_load_providers_skips_non_provider_files(tmp_path):
    """Files without 'prefix' field are skipped."""
    legacy = {"_type": "giga-chat-model", "max_tokens": 100}
    (tmp_path / "legacy.json").write_text(json.dumps(legacy))
    h = LlmHub()
    h.load_providers(str(tmp_path))
    assert len(h.providers) == 0


def test_load_providers_rejects_duplicate_prefix(tmp_path):
    p1 = {"prefix": "dup", "base_url": "http://a"}
    p2 = {"prefix": "dup", "base_url": "http://b"}
    (tmp_path / "a.json").write_text(json.dumps(p1))
    (tmp_path / "b.json").write_text(json.dumps(p2))
    h = LlmHub()
    h.load_providers(str(tmp_path))
    assert len(h.providers) == 1  # first one wins


def test_load_providers_missing_directory():
    h = LlmHub()
    h.load_providers("/nonexistent/path")
    assert len(h.providers) == 0


def test_load_providers_invalid_json(tmp_path):
    (tmp_path / "bad.json").write_text("not json")
    h = LlmHub()
    h.load_providers(str(tmp_path))
    assert len(h.providers) == 0


def test_load_providers_env_var_expansion(tmp_path):
    p = {"prefix": "envtest", "base_url": "${TEST_LLM_HUB_URL}"}
    (tmp_path / "env.json").write_text(json.dumps(p))
    os.environ["TEST_LLM_HUB_URL"] = "http://expanded.api"
    try:
        h = LlmHub()
        h.load_providers(str(tmp_path))
        assert h.providers["envtest"].config.base_url == "http://expanded.api"
    finally:
        del os.environ["TEST_LLM_HUB_URL"]


def test_resolve_model_with_known_prefix(hub):
    result = hub.resolve_model("alpha/model-a")
    assert result is not None
    provider, model = result
    assert provider.config.prefix == "alpha"
    assert model == "model-a"


def test_resolve_model_unknown_prefix(hub):
    result = hub.resolve_model("unknown/model")
    assert result is None


def test_resolve_model_no_slash(hub):
    result = hub.resolve_model("plain-model-name")
    assert result is None


def test_resolve_model_with_slash_in_model_name(hub):
    """Model names like 'meta-llama/Llama-3' should fall back to None if prefix is unknown."""
    result = hub.resolve_model("meta-llama/Llama-3")
    assert result is None


def test_resolve_model_multi_slash(hub):
    """Only splits on first /."""
    result = hub.resolve_model("alpha/deep/model")
    assert result is not None
    provider, model = result
    assert model == "deep/model"


def test_get_merged_models(hub):
    result = hub.get_merged_models()
    assert result["object"] == "list"
    ids = [m["id"] for m in result["data"]]
    assert "alpha/model-a" in ids
    assert "alpha/model-b" in ids
    assert "beta/model-x" in ids
    for m in result["data"]:
        assert m["object"] == "model"
        assert "owned_by" in m


def test_get_merged_models_empty():
    h = LlmHub()
    result = h.get_merged_models()
    assert result == {"object": "list", "data": []}


@pytest.mark.asyncio
async def test_provider_get_auth_headers_apikey():
    config = LlmProviderConfig(
        prefix="test",
        base_url="http://test",
        auth=AuthConfig(type=AuthType.APIKEY, api_key="mykey"),
    )
    provider = LlmProvider(config)
    headers = await provider.get_auth_headers()
    assert headers == {"X-API-KEY": "mykey"}


@pytest.mark.asyncio
async def test_provider_get_auth_headers_none():
    config = LlmProviderConfig(
        prefix="test",
        base_url="http://test",
        auth=AuthConfig(type=AuthType.NONE),
    )
    provider = LlmProvider(config)
    headers = await provider.get_auth_headers()
    assert headers == {}


@pytest.mark.asyncio
async def test_provider_get_auth_headers_bearer():
    config = LlmProviderConfig(
        prefix="test",
        base_url="http://test",
        auth=AuthConfig(type=AuthType.BEARER, oidc_url="http://auth", credentials="cred", scope="S"),
    )
    provider = LlmProvider(config)
    await provider.startup()
    provider.token_manager = MagicMock()
    provider.token_manager.get_token = AsyncMock(return_value="bearer-tok")
    headers = await provider.get_auth_headers()
    assert headers == {"Authorization": "Bearer bearer-tok"}
    await provider.shutdown()


@pytest.mark.asyncio
async def test_hub_startup_shutdown(provider_dir):
    hub = LlmHub()
    hub.load_providers(provider_dir)
    await hub.startup()
    for p in hub.providers.values():
        assert p.client is not None
    await hub.shutdown()
