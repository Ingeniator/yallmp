import pytest
import json
import os
from unittest.mock import AsyncMock, MagicMock

from app.services.llm_hub import LlmHub, LlmProvider
from app.schemas.provider import LlmProviderConfig, AuthType, AuthConfig, AliasEntry


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


def test_resolve_model_unknown_prefix_without_default(hub):
    assert hub.default_prefix is None
    assert hub.resolve_model("unknown/model") is None


def test_resolve_model_no_slash_without_default(hub):
    assert hub.resolve_model("plain-model-name") is None


def test_resolve_model_unprefixed_from_models_list(hub):
    """An unprefixed name listed in a provider's ``models`` goes there unchanged."""
    provider, model = hub.resolve_model("model-x")
    assert provider.config.prefix == "beta"
    assert model == "model-x"


@pytest.fixture
def hub_with_default(provider_dir):
    path = os.path.join(provider_dir, "gamma.json")
    with open(path, "w") as f:
        json.dump({"prefix": "gamma", "base_url": "http://gamma", "default": True}, f)
    h = LlmHub()
    h.load_providers(provider_dir)
    return h


def test_default_provider_loaded(hub_with_default):
    assert hub_with_default.default_prefix == "gamma"
    assert hub_with_default.default_provider.config.prefix == "gamma"


def test_resolve_model_no_slash_goes_to_default(hub_with_default):
    provider, model = hub_with_default.resolve_model("plain-model-name")
    assert provider.config.prefix == "gamma"
    assert model == "plain-model-name"


def test_resolve_model_unknown_prefix_goes_to_default_unchanged(hub_with_default):
    """'meta-llama/Llama-3' is a model name, not a prefix: sent whole to the default."""
    provider, model = hub_with_default.resolve_model("meta-llama/Llama-3")
    assert provider.config.prefix == "gamma"
    assert model == "meta-llama/Llama-3"


def test_resolve_model_models_list_beats_default(hub_with_default):
    provider, model = hub_with_default.resolve_model("model-a")
    assert provider.config.prefix == "alpha"
    assert model == "model-a"


def test_resolve_model_known_prefix_beats_default(hub_with_default):
    provider, model = hub_with_default.resolve_model("beta/anything")
    assert provider.config.prefix == "beta"
    assert model == "anything"


def test_multiple_defaults_first_wins(tmp_path):
    for name in ("a", "b"):
        (tmp_path / f"{name}.json").write_text(
            json.dumps({"prefix": name, "base_url": f"http://{name}", "default": True})
        )
    h = LlmHub()
    h.load_providers(str(tmp_path))
    assert h.default_prefix == "a"
    assert set(h.providers) == {"a", "b"}


def test_resolve_model_multi_slash(hub):
    """Only splits on first /."""
    result = hub.resolve_model("alpha/deep/model")
    assert result is not None
    provider, model = result
    assert model == "deep/model"


@pytest.mark.asyncio
async def test_get_merged_models(hub):
    result = await hub.get_merged_models()
    assert result["object"] == "list"
    ids = [m["id"] for m in result["data"]]
    assert "alpha/model-a" in ids
    assert "alpha/model-b" in ids
    assert "beta/model-x" in ids
    for m in result["data"]:
        assert m["object"] == "model"
        assert "owned_by" in m


@pytest.mark.asyncio
async def test_get_merged_models_hides_providers_outside_allow_list(hub):
    hub.providers["beta"].config.allow_list = ["team1"]

    denied = await hub.get_merged_models(request_headers={"x-group-id": "team2"})
    denied_ids = [m["id"] for m in denied["data"]]
    assert "beta/model-x" not in denied_ids
    assert "alpha/model-a" in denied_ids

    allowed = await hub.get_merged_models(request_headers={"x-group-id": "team1/user"})
    assert "beta/model-x" in [m["id"] for m in allowed["data"]]

    anonymous = await hub.get_merged_models()
    assert "beta/model-x" not in [m["id"] for m in anonymous["data"]]


@pytest.mark.asyncio
async def test_get_merged_models_empty():
    h = LlmHub()
    result = await h.get_merged_models()
    assert result == {"object": "list", "data": []}


def _models_response(entries, status=200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = {"object": "list", "data": entries}
    if status >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status}")
    else:
        resp.raise_for_status.return_value = None
    return resp


def _dynamic_hub(client):
    h = LlmHub()
    provider = LlmProvider(
        LlmProviderConfig(
            prefix="dyn",
            base_url="http://dyn/",
            auth=AuthConfig(type=AuthType.APIKEY, api_key="k"),
        )
    )
    provider.client = client
    h.providers["dyn"] = provider
    return h


@pytest.mark.asyncio
async def test_get_merged_models_dynamic_keeps_attributes():
    client = MagicMock()
    client.get = AsyncMock(
        return_value=_models_response(
            [{"id": "m1", "object": "model", "owned_by": "org", "context_length": 8192}]
        )
    )
    h = _dynamic_hub(client)

    result = await h.get_merged_models()

    assert result["data"] == [
        {"id": "dyn/m1", "object": "model", "owned_by": "org", "context_length": 8192}
    ]
    assert client.get.call_args.args[0] == "http://dyn/v1/models"
    assert client.get.call_args.kwargs["headers"] == {"X-API-KEY": "k"}


@pytest.mark.asyncio
async def test_get_merged_models_dynamic_is_cached():
    client = MagicMock()
    client.get = AsyncMock(return_value=_models_response([{"id": "m1"}]))
    h = _dynamic_hub(client)

    await h.get_merged_models()
    await h.get_merged_models()

    assert client.get.await_count == 1


@pytest.mark.asyncio
async def test_get_merged_models_cache_expires(monkeypatch):
    import app.services.llm_hub as llm_hub_module

    client = MagicMock()
    client.get = AsyncMock(return_value=_models_response([{"id": "m1"}]))
    h = _dynamic_hub(client)

    now = [1000.0]
    monkeypatch.setattr(llm_hub_module.time, "monotonic", lambda: now[0])
    await h.get_merged_models()
    now[0] += llm_hub_module._MODELS_TTL + 1
    await h.get_merged_models()

    assert client.get.await_count == 2


@pytest.mark.asyncio
async def test_get_merged_models_provider_failure_serves_stale(monkeypatch):
    import app.services.llm_hub as llm_hub_module

    client = MagicMock()
    client.get = AsyncMock(return_value=_models_response([{"id": "m1"}]))
    h = _dynamic_hub(client)

    now = [1000.0]
    monkeypatch.setattr(llm_hub_module.time, "monotonic", lambda: now[0])
    await h.get_merged_models()

    now[0] += llm_hub_module._MODELS_TTL + 1
    client.get = AsyncMock(side_effect=Exception("connect error"))
    result = await h.get_merged_models()
    assert [m["id"] for m in result["data"]] == ["dyn/m1"]

    # Failed refresh is not retried until the retry window passes
    await h.get_merged_models()
    assert client.get.await_count == 1


@pytest.mark.asyncio
async def test_get_merged_models_provider_unreachable_returns_rest(hub):
    client = MagicMock()
    client.get = AsyncMock(side_effect=Exception("connect error"))
    hub.providers["dyn"] = _dynamic_hub(client).providers["dyn"]

    result = await hub.get_merged_models()

    ids = [m["id"] for m in result["data"]]
    assert "alpha/model-a" in ids
    assert not any(i.startswith("dyn/") for i in ids)


@pytest.mark.asyncio
async def test_get_merged_models_skips_provider_without_client():
    h = LlmHub()
    h.providers["dyn"] = LlmProvider(
        LlmProviderConfig(
            prefix="dyn", base_url="http://dyn", auth=AuthConfig(type=AuthType.NONE)
        )
    )
    result = await h.get_merged_models()
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


def _passthrough_provider(headers=None):
    return LlmProvider(
        LlmProviderConfig(
            prefix="test",
            base_url="http://test",
            auth=AuthConfig(type=AuthType.PASSTHROUGH, headers=headers),
        )
    )


@pytest.mark.asyncio
async def test_provider_get_auth_headers_passthrough_default():
    provider = _passthrough_provider()
    headers = await provider.get_auth_headers(
        {"Authorization": "Bearer gateway-tok", "X-API-KEY": "llm-key", "Cookie": "c"}
    )
    # The client's Authorization (gateway token) must never be forwarded
    assert headers == {"X-API-KEY": "llm-key"}
    assert await provider.get_auth_headers({}) == {}
    assert await provider.get_auth_headers() == {}


@pytest.mark.asyncio
async def test_provider_get_auth_headers_passthrough_template():
    from starlette.datastructures import Headers

    provider = _passthrough_provider({"Authorization": "Bearer ${headers.X-API-KEY}"})
    request_headers = Headers(
        {"authorization": "Bearer gateway-tok", "x-api-key": "llm-key"}
    )
    headers = await provider.get_auth_headers(request_headers)
    assert headers == {"Authorization": "Bearer llm-key"}


@pytest.mark.asyncio
async def test_provider_get_auth_headers_passthrough_missing_source_header():
    provider = _passthrough_provider(
        {"Authorization": "Bearer ${headers.X-API-KEY}", "X-Static": "fixed"}
    )
    headers = await provider.get_auth_headers({"Authorization": "Bearer gateway-tok"})
    assert headers == {"X-Static": "fixed"}


def test_load_passthrough_provider_from_json(tmp_path):
    cfg = {
        "prefix": "pt",
        "base_url": "http://pt",
        "auth": {
            "type": "PASSTHROUGH",
            "headers": {"Authorization": "Bearer ${headers.X-API-KEY}"},
        },
        "models": [],
    }
    (tmp_path / "pt.json").write_text(json.dumps(cfg))
    hub = LlmHub()
    hub.load_providers(str(tmp_path))
    auth = hub.providers["pt"].config.auth
    assert auth.type == AuthType.PASSTHROUGH
    assert auth.headers == {"Authorization": "Bearer ${headers.X-API-KEY}"}


def _passthrough_hub(client):
    h = LlmHub()
    provider = _passthrough_provider({"Authorization": "Bearer ${headers.X-API-KEY}"})
    provider.client = client
    h.providers["test"] = provider
    return h


@pytest.mark.asyncio
async def test_get_merged_models_passthrough_without_key_skips_fetch():
    client = MagicMock()
    client.get = AsyncMock(return_value=_models_response([{"id": "m1"}]))
    h = _passthrough_hub(client)

    result = await h.get_merged_models(request_headers={})

    assert result["data"] == []
    client.get.assert_not_awaited()
    assert h._models_cache == {}


@pytest.mark.asyncio
async def test_get_merged_models_passthrough_cached_per_key():
    lists = {
        "Bearer key-a": [{"id": "a-model"}],
        "Bearer key-b": [{"id": "b-model"}],
    }
    client = MagicMock()
    client.get = AsyncMock(
        side_effect=lambda url, headers, timeout: _models_response(
            lists[headers["Authorization"]]
        )
    )
    h = _passthrough_hub(client)

    a1 = await h.get_merged_models(request_headers={"X-API-KEY": "key-a"})
    b1 = await h.get_merged_models(request_headers={"X-API-KEY": "key-b"})
    a2 = await h.get_merged_models(request_headers={"X-API-KEY": "key-a"})
    none = await h.get_merged_models(request_headers={})

    assert [m["id"] for m in a1["data"]] == ["test/a-model"]
    assert [m["id"] for m in b1["data"]] == ["test/b-model"]
    assert a2 == a1
    assert none["data"] == []
    assert client.get.await_count == 2
    # Raw credentials never appear in cache keys
    assert all("key-" not in k for k in h._models_cache)


@pytest.mark.asyncio
async def test_provider_get_auth_headers_bearer_static_key():
    config = LlmProviderConfig(
        prefix="test",
        base_url="http://test",
        auth=AuthConfig(type=AuthType.BEARER, api_key="static-key"),
    )
    provider = LlmProvider(config)
    await provider.startup()
    assert provider.token_manager is None
    headers = await provider.get_auth_headers({"X-API-KEY": "client"})
    assert headers == {"Authorization": "Bearer static-key"}
    await provider.shutdown()


@pytest.mark.asyncio
async def test_provider_get_auth_headers_oidc():
    config = LlmProviderConfig(
        prefix="test",
        base_url="http://test",
        auth=AuthConfig(type=AuthType.OIDC, oidc_url="http://auth", credentials="cred", scope="S"),
    )
    provider = LlmProvider(config)
    await provider.startup()
    provider.token_manager = MagicMock()
    provider.token_manager.get_token = AsyncMock(return_value="bearer-tok")
    headers = await provider.get_auth_headers()
    assert headers == {"Authorization": "Bearer bearer-tok"}
    await provider.shutdown()


@pytest.fixture
def hub_with_aliases(provider_dir):
    aliases = {
        "smart": {"target": "alpha/model-a", "fallback": "beta/model-x"},
        "fast":  {"target": "alpha/model-b"},
    }
    import pathlib
    (pathlib.Path(provider_dir) / "aliases.json").write_text(json.dumps(aliases))
    h = LlmHub()
    h.load_providers(provider_dir)
    return h


def test_load_aliases(hub_with_aliases):
    assert "smart" in hub_with_aliases.aliases
    assert "fast" in hub_with_aliases.aliases
    assert len(hub_with_aliases.aliases) == 2


def test_alias_target_and_fallback(hub_with_aliases):
    entry = hub_with_aliases.aliases["smart"]
    assert entry.target == "alpha/model-a"
    assert entry.fallback == "beta/model-x"


def test_alias_no_fallback(hub_with_aliases):
    entry = hub_with_aliases.aliases["fast"]
    assert entry.target == "alpha/model-b"
    assert entry.fallback is None


def test_resolve_alias_known(hub_with_aliases):
    entry = hub_with_aliases.resolve_alias("smart")
    assert entry is not None
    assert entry.target == "alpha/model-a"


def test_resolve_alias_unknown(hub_with_aliases):
    assert hub_with_aliases.resolve_alias("unknown-alias") is None


def test_aliases_missing_file(provider_dir):
    """Hub with no aliases.json starts with empty aliases dict."""
    h = LlmHub()
    h.load_providers(provider_dir)
    assert h.aliases == {}


def test_aliases_invalid_json(tmp_path):
    p = {"prefix": "alpha", "base_url": "http://alpha.api/v1", "models": ["m"]}
    (tmp_path / "alpha.json").write_text(json.dumps(p))
    (tmp_path / "aliases.json").write_text("not json")
    h = LlmHub()
    h.load_providers(str(tmp_path))
    assert h.aliases == {}


def test_example_aliases_file_loads():
    """The checked-in data/llm_hub/aliases.json must parse without errors."""
    import pathlib
    hub_dir = pathlib.Path(__file__).parent.parent / "data" / "llm_hub"
    h = LlmHub()
    h.load_providers(str(hub_dir))
    assert len(h.aliases) > 0
    for name, entry in h.aliases.items():
        assert "/" in entry.target, f"alias '{name}' target must be prefix/model"
        if entry.fallback:
            assert "/" in entry.fallback, f"alias '{name}' fallback must be prefix/model"


@pytest.mark.asyncio
async def test_hub_startup_shutdown(provider_dir):
    hub = LlmHub()
    hub.load_providers(provider_dir)
    await hub.startup()
    for p in hub.providers.values():
        assert p.client is not None
    await hub.shutdown()


def test_load_providers_skips_unset_env_var(tmp_path):
    p = {
        "prefix": "unset",
        "base_url": "http://x",
        "auth": {"type": "BEARER", "api_key": "${TEST_LLM_HUB_DEFINITELY_UNSET}"},
    }
    (tmp_path / "unset.json").write_text(json.dumps(p))
    os.environ.pop("TEST_LLM_HUB_DEFINITELY_UNSET", None)
    h = LlmHub()
    h.load_providers(str(tmp_path))
    assert "unset" not in h.providers


def test_load_providers_keeps_passthrough_placeholders(tmp_path):
    p = {
        "prefix": "pt",
        "base_url": "http://x",
        "auth": {"type": "PASSTHROUGH", "headers": {"Authorization": "Bearer ${headers.X-API-KEY}"}},
    }
    (tmp_path / "pt.json").write_text(json.dumps(p))
    h = LlmHub()
    h.load_providers(str(tmp_path))
    assert "pt" in h.providers


@pytest.mark.asyncio
async def test_models_cache_prunes_long_expired_entries(monkeypatch):
    import app.services.llm_hub as llm_hub_module

    h = LlmHub()
    now = [1000.0]
    monkeypatch.setattr(llm_hub_module.time, "monotonic", lambda: now[0])
    fetch = AsyncMock(return_value=[{"id": "m"}])
    await h._cached_models("p#old", fetch)
    assert "p#old" in h._models_cache

    # Recently expired: kept as stale fallback
    now[0] += llm_hub_module._MODELS_TTL + 1
    await h.get_merged_models()
    assert "p#old" in h._models_cache

    now[0] += llm_hub_module._MODELS_PRUNE_AFTER
    await h.get_merged_models()
    assert "p#old" not in h._models_cache
    assert "p#old" not in h._models_locks


def test_load_providers_rejects_prefix_with_slash(tmp_path):
    (tmp_path / "p.json").write_text(json.dumps({"prefix": "a/b", "base_url": "http://x"}))
    h = LlmHub()
    h.load_providers(str(tmp_path))
    assert h.providers == {}


def test_load_providers_allows_former_reserved_prefixes(tmp_path):
    """With the /llm/@<prefix>/ marker, prefixes like 'v1' can't clash with API paths."""
    (tmp_path / "p.json").write_text(json.dumps({"prefix": "v1", "base_url": "http://x"}))
    h = LlmHub()
    h.load_providers(str(tmp_path))
    assert "v1" in h.providers
