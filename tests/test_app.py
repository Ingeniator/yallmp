import json
from unittest.mock import patch, AsyncMock, MagicMock

import pytest
from starlette.testclient import TestClient
from fastapi.responses import JSONResponse


def test_health_all_disabled():
    with patch("app.core.app.settings") as mock_settings:
        mock_settings.app_name = "TestApp"
        mock_settings.debug = False
        mock_settings.root_path = ""
        mock_settings.allowed_origins = ["*"]
        mock_settings.proxy_enabled = False
        mock_settings.prompt_hub_enabled = False
        mock_settings.chain_hub_enabled = False
        mock_settings.llm_hub_enabled = False
        mock_settings.dashboard_enabled = False
        mock_settings.search_hub_enabled = False
        mock_settings.tracing_enabled = False
        mock_settings.billing_enabled = False
        mock_settings.version = "0.0.1-test"

        from app.core.app import create_app
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.0.1-test"
        assert data["components"]["proxy"] == "disabled"


def test_health_all_enabled():
    with patch("app.core.app.settings") as mock_settings:
        mock_settings.app_name = "TestApp"
        mock_settings.debug = False
        mock_settings.root_path = ""
        mock_settings.allowed_origins = ["*"]
        mock_settings.proxy_enabled = True
        mock_settings.prompt_hub_enabled = True
        mock_settings.chain_hub_enabled = True
        mock_settings.llm_hub_enabled = True
        mock_settings.dashboard_enabled = True
        mock_settings.search_hub_enabled = False
        mock_settings.tracing_enabled = False
        mock_settings.billing_enabled = False
        mock_settings.version = "0.0.1-test"

        from app.core.app import create_app
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "ok"
        assert all(v in ("ok", "disabled") for v in data["components"].values())


def test_metrics_endpoint():
    with patch("app.core.app.settings") as mock_settings:
        mock_settings.app_name = "TestApp"
        mock_settings.debug = False
        mock_settings.root_path = ""
        mock_settings.allowed_origins = ["*"]
        mock_settings.proxy_enabled = False
        mock_settings.prompt_hub_enabled = False
        mock_settings.chain_hub_enabled = False
        mock_settings.llm_hub_enabled = False
        mock_settings.dashboard_enabled = False
        mock_settings.version = "0.0.1-test"

        from app.core.app import create_app
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/metrics")
        assert resp.status_code == 200


def test_lifespan_no_proxy():
    """Lifespan runs startup/shutdown without proxy."""
    with patch("app.core.app.settings") as mock_settings:
        mock_settings.app_name = "TestApp"
        mock_settings.debug = False
        mock_settings.root_path = ""
        mock_settings.allowed_origins = ["*"]
        mock_settings.proxy_enabled = False
        mock_settings.prompt_hub_enabled = False
        mock_settings.chain_hub_enabled = False
        mock_settings.llm_hub_enabled = False
        mock_settings.dashboard_enabled = False
        mock_settings.search_hub_enabled = False
        mock_settings.tracing_enabled = False
        mock_settings.billing_enabled = False
        mock_settings.version = "0.0.1-test"

        from app.core.app import create_app
        app = create_app()

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            assert app.state.llm_hub.providers == {}


# --- Feature-flag route handler tests ---


def _mock_settings(**overrides):
    """Return a patch context for app.core.app.settings with given feature flags."""
    defaults = dict(
        app_name="TestApp", debug=False, root_path="", allowed_origins=["*"],
        proxy_enabled=False, prompt_hub_enabled=False, chain_hub_enabled=False,
        llm_hub_enabled=False, dashboard_enabled=False,
        search_hub_enabled=False, tracing_enabled=False, billing_enabled=False,
        debug_network_check=False, version="0.0.1-test",
    )
    defaults.update(overrides)

    class FakeSettings:
        pass

    s = FakeSettings()
    for k, v in defaults.items():
        setattr(s, k, v)
    return s


# --- /llm routing through LLM hub providers ---


@pytest.fixture
def hub_dir(tmp_path, monkeypatch):
    """Provider dir with a prefixed provider, a default one and an alias."""
    import app.services.llm_hub as llm_hub_module

    (tmp_path / "alpha.json").write_text(json.dumps(
        {"prefix": "alpha", "base_url": "http://alpha", "models": ["alpha-model"]}
    ))
    (tmp_path / "local.json").write_text(json.dumps(
        {"prefix": "local", "base_url": "http://local", "default": True}
    ))
    (tmp_path / "aliases.json").write_text(json.dumps(
        {
            "smart": {"target": "alpha/big"},
            "both": {"target": "alpha/big", "fallback": "local/small"},
            "local-first": {"target": "local/small", "fallback": "alpha/big"},
            "alpha-pair": {"target": "alpha/big", "fallback": "alpha/medium"},
        }
    ))
    monkeypatch.setattr(llm_hub_module.settings, "llm_hub_directory", str(tmp_path))
    return tmp_path


def _proxy_client(mock_proxy=None, mock_version=None):
    mock_proxy = mock_proxy or AsyncMock(
        return_value=JSONResponse(content={"choices": []}, status_code=200)
    )
    patches = [
        patch("app.core.app.settings", _mock_settings(proxy_enabled=True)),
        patch("app.core.proxy.proxy_request_to_provider", mock_proxy),
        patch("app.core.proxy.get_model_version", mock_version or AsyncMock()),
    ]
    return patches, mock_proxy


def _call(patches, method, path, **kw):
    from contextlib import ExitStack
    from app.core.app import create_app

    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        app = create_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            return client.request(method, path, **kw)


def _routed(mock_proxy):
    kw = mock_proxy.call_args.kwargs
    return kw["provider"].config.prefix, kw["original_model"], kw["stripped_model"]


def test_llm_route_prefixed_model_strips_prefix(hub_dir):
    patches, mock_proxy = _proxy_client()
    resp = _call(patches, "POST", "/llm/v1/chat/completions", json={"model": "alpha/x"})
    assert resp.status_code == 200
    assert _routed(mock_proxy) == ("alpha", "alpha/x", "x")


def test_llm_route_unprefixed_model_goes_to_default_unchanged(hub_dir):
    patches, mock_proxy = _proxy_client()
    _call(patches, "POST", "/llm/v1/chat/completions", json={"model": "GigaChat"})
    assert _routed(mock_proxy) == ("local", "GigaChat", "GigaChat")


def test_llm_route_unknown_prefix_goes_to_default_unchanged(hub_dir):
    patches, mock_proxy = _proxy_client()
    _call(patches, "POST", "/llm/v1/chat/completions", json={"model": "meta/llama"})
    assert _routed(mock_proxy) == ("local", "meta/llama", "meta/llama")


def test_llm_route_models_list_match(hub_dir):
    patches, mock_proxy = _proxy_client()
    _call(patches, "POST", "/llm/v1/chat/completions", json={"model": "alpha-model"})
    assert _routed(mock_proxy) == ("alpha", "alpha-model", "alpha-model")


def test_llm_route_alias(hub_dir):
    patches, mock_proxy = _proxy_client()
    _call(patches, "POST", "/llm/v1/chat/completions", json={"model": "smart"})
    assert _routed(mock_proxy) == ("alpha", "smart", "big")


def test_llm_route_get_goes_to_default(hub_dir):
    patches, mock_proxy = _proxy_client()
    resp = _call(patches, "GET", "/llm/v1/files")
    assert resp.status_code == 200
    assert _routed(mock_proxy) == ("local", "", "")


def test_llm_route_unknown_model_without_default_is_404(hub_dir):
    (hub_dir / "local.json").unlink()
    patches, mock_proxy = _proxy_client()
    resp = _call(patches, "POST", "/llm/v1/chat/completions", json={"model": "GigaChat"})
    assert resp.status_code == 404
    assert "GigaChat" in resp.json()["error"]
    mock_proxy.assert_not_awaited()


def test_llm_route_no_model_without_default_is_404(hub_dir):
    (hub_dir / "local.json").unlink()
    patches, mock_proxy = _proxy_client()
    resp = _call(patches, "GET", "/llm/v1/files")
    assert resp.status_code == 404
    mock_proxy.assert_not_awaited()


def test_llm_version_route_uses_resolved_provider(hub_dir):
    mock_version = AsyncMock(return_value={"version": "GigaChat-9b-128k-base:3.0"})
    patches, _ = _proxy_client(mock_version=mock_version)
    resp = _call(patches, "GET", "/llm/version")
    assert resp.status_code == 200
    assert resp.json() == {"version": "GigaChat-9b-128k-base:3.0"}
    model, provider = mock_version.call_args.args[:2]
    assert model == "GigaChat"
    assert provider.config.prefix == "local"


def test_llm_models_route_lists_hub_models(hub_dir):
    patches, _ = _proxy_client()
    with patch("app.services.llm_hub._fetch_model_list", AsyncMock(return_value=[{"id": "m1"}])):
        resp = _call(patches, "GET", "/llm/v1/models")
    ids = {m["id"] for m in resp.json()["data"]}
    assert ids == {"alpha/alpha-model", "local/m1"}


def _routed_path(mock_proxy):
    kw = mock_proxy.call_args.kwargs
    return kw["provider"].config.prefix, kw["path"]


def test_llm_path_prefix_pins_provider_for_upload(hub_dir):
    patches, mock_proxy = _proxy_client()
    resp = _call(patches, "POST", "/llm/@alpha/v1/files",
                 files={"file": ("t.jsonl", b"{}")}, data={"purpose": "fine-tune"})
    assert resp.status_code == 200
    assert _routed_path(mock_proxy) == ("alpha", "v1/files")


def test_llm_path_prefix_pins_provider_for_get(hub_dir):
    patches, mock_proxy = _proxy_client()
    _call(patches, "GET", "/llm/@alpha/v1/fine_tuning/jobs/ftjob-1")
    assert _routed_path(mock_proxy) == ("alpha", "v1/fine_tuning/jobs/ftjob-1")


def test_llm_pinned_unprefixed_model_unchanged(hub_dir):
    patches, mock_proxy = _proxy_client()
    _call(patches, "POST", "/llm/@alpha/v1/fine_tuning/jobs", json={"model": "gpt-x"})
    assert _routed(mock_proxy) == ("alpha", "gpt-x", "gpt-x")


def test_llm_pinned_alias_uses_entry_on_pinned_provider(hub_dir):
    patches, mock_proxy = _proxy_client()
    _call(patches, "POST", "/llm/@alpha/v1/chat/completions", json={"model": "smart"})
    assert _routed(mock_proxy) == ("alpha", "smart", "big")


def test_llm_pinned_alias_uses_fallback_entry_when_target_elsewhere(hub_dir):
    patches, mock_proxy = _proxy_client()
    _call(patches, "POST", "/llm/@alpha/v1/chat/completions", json={"model": "local-first"})
    assert _routed(mock_proxy) == ("alpha", "local-first", "big")


def test_llm_pinned_alias_never_falls_back_to_other_provider(hub_dir):
    mock_proxy = AsyncMock(return_value=JSONResponse(content={}, status_code=500))
    patches, _ = _proxy_client(mock_proxy=mock_proxy)
    resp = _call(patches, "POST", "/llm/@alpha/v1/chat/completions", json={"model": "both"})
    assert resp.status_code == 500
    assert mock_proxy.await_count == 1
    assert _routed(mock_proxy) == ("alpha", "both", "big")


def test_llm_pinned_alias_falls_back_within_provider(hub_dir):
    mock_proxy = AsyncMock(side_effect=[
        JSONResponse(content={}, status_code=500),
        JSONResponse(content={"ok": True}, status_code=200),
    ])
    patches, _ = _proxy_client(mock_proxy=mock_proxy)
    resp = _call(patches, "POST", "/llm/@alpha/v1/chat/completions", json={"model": "alpha-pair"})
    assert resp.status_code == 200
    assert _routed(mock_proxy) == ("alpha", "alpha-pair", "medium")


def test_llm_pinned_alias_without_entry_on_provider_sent_unchanged(hub_dir):
    """No entry on the pinned provider: the name may be a real upstream model."""
    patches, mock_proxy = _proxy_client()
    _call(patches, "POST", "/llm/@local/v1/chat/completions", json={"model": "smart"})
    assert _routed(mock_proxy) == ("local", "smart", "smart")


def test_llm_header_pin_alias_same_rules(hub_dir):
    patches, mock_proxy = _proxy_client()
    _call(patches, "POST", "/llm/v1/chat/completions", json={"model": "local-first"},
          headers={"X-Provider": "alpha"})
    assert _routed(mock_proxy) == ("alpha", "local-first", "big")


def test_llm_unpinned_alias_falls_back_across_providers(hub_dir):
    mock_proxy = AsyncMock(side_effect=[
        JSONResponse(content={}, status_code=500),
        JSONResponse(content={"ok": True}, status_code=200),
    ])
    patches, _ = _proxy_client(mock_proxy=mock_proxy)
    resp = _call(patches, "POST", "/llm/v1/chat/completions", json={"model": "both"})
    assert resp.status_code == 200
    assert _routed(mock_proxy) == ("local", "both", "small")


def test_llm_path_prefix_strips_own_model_prefix(hub_dir):
    patches, mock_proxy = _proxy_client()
    _call(patches, "POST", "/llm/@alpha/v1/chat/completions", json={"model": "alpha/x"})
    assert _routed(mock_proxy) == ("alpha", "alpha/x", "x")
    assert _routed_path(mock_proxy) == ("alpha", "v1/chat/completions")


def test_llm_provider_name_without_marker_is_not_a_pin(hub_dir):
    """/llm/alpha/... is an upstream path for the default provider, not a pin."""
    patches, mock_proxy = _proxy_client()
    _call(patches, "GET", "/llm/alpha/v1/files")
    assert _routed_path(mock_proxy) == ("local", "alpha/v1/files")


def test_llm_unknown_path_pin_is_404(hub_dir):
    patches, mock_proxy = _proxy_client()
    resp = _call(patches, "GET", "/llm/@nope/v1/files")
    assert resp.status_code == 404
    assert "nope" in resp.json()["error"]
    mock_proxy.assert_not_awaited()


def test_llm_provider_header_pins_provider(hub_dir):
    patches, mock_proxy = _proxy_client()
    _call(patches, "GET", "/llm/v1/files", headers={"X-Provider": "alpha"})
    assert _routed_path(mock_proxy) == ("alpha", "v1/files")


def test_llm_provider_header_unknown_is_404(hub_dir):
    patches, mock_proxy = _proxy_client()
    resp = _call(patches, "GET", "/llm/v1/files", headers={"X-Provider": "nope"})
    assert resp.status_code == 404
    mock_proxy.assert_not_awaited()


def test_llm_provider_header_conflicting_with_path_is_400(hub_dir):
    patches, mock_proxy = _proxy_client()
    resp = _call(patches, "GET", "/llm/@alpha/v1/files", headers={"X-Provider": "local"})
    assert resp.status_code == 400
    mock_proxy.assert_not_awaited()


def test_llm_provider_header_matching_path_is_ok(hub_dir):
    patches, mock_proxy = _proxy_client()
    _call(patches, "GET", "/llm/@alpha/v1/files", headers={"X-Provider": "alpha"})
    assert _routed_path(mock_proxy) == ("alpha", "v1/files")


def test_llm_models_route_header_pin_proxies_provider_list(hub_dir):
    patches, mock_proxy = _proxy_client()
    resp = _call(patches, "GET", "/llm/v1/models", headers={"X-Provider": "alpha"})
    assert resp.status_code == 200
    assert _routed_path(mock_proxy) == ("alpha", "v1/models")


def test_llm_models_route_header_unknown_is_404(hub_dir):
    patches, mock_proxy = _proxy_client()
    resp = _call(patches, "GET", "/llm/v1/models", headers={"X-Provider": "nope"})
    assert resp.status_code == 404
    mock_proxy.assert_not_awaited()


def test_llm_version_route_header_pin(hub_dir):
    mock_version = AsyncMock(return_value={"version": "v"})
    patches, _ = _proxy_client(mock_version=mock_version)
    _call(patches, "GET", "/llm/version?model_name=alpha/GigaChat", headers={"X-Provider": "alpha"})
    model, provider = mock_version.call_args.args[:2]
    assert (provider.config.prefix, model) == ("alpha", "GigaChat")


def test_llm_version_route_header_unknown_is_404(hub_dir):
    mock_version = AsyncMock()
    patches, _ = _proxy_client(mock_version=mock_version)
    resp = _call(patches, "GET", "/llm/version", headers={"X-Provider": "nope"})
    assert resp.status_code == 404
    mock_version.assert_not_awaited()


def test_health_proxy_degraded_without_providers(tmp_path, monkeypatch):
    import app.services.llm_hub as llm_hub_module

    monkeypatch.setattr(llm_hub_module.settings, "llm_hub_directory", str(tmp_path))
    patches, _ = _proxy_client()
    resp = _call(patches, "GET", "/health")
    data = resp.json()
    assert data["components"]["proxy"] == "degraded"
    assert data["details"]["proxy"] == {"providers": "none loaded"}


def test_prompt_hub_get_prompts_route():
    """GET /prompts invokes promptStore.get_prompts."""
    mock_prompt_store = MagicMock()
    mock_prompt_store.get_prompts = AsyncMock(return_value={"greeting": {"input_variables": ["name"]}})

    with patch("app.core.app.settings", _mock_settings(prompt_hub_enabled=True)), \
         patch("app.services.prompt_manager.get_prompt_store", return_value=mock_prompt_store):

        from app.core.app import create_app
        app = create_app()

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/prompts")

    assert resp.status_code == 200
    assert "greeting" in resp.json()
    mock_prompt_store.get_prompts.assert_awaited_once()


def test_prompt_hub_format_prompt_route():
    """POST /prompt/format/{name} invokes promptStore.format_prompt."""
    mock_prompt_store = MagicMock()
    mock_prompt_store.format_prompt = AsyncMock(return_value="Hello, World!")

    with patch("app.core.app.settings", _mock_settings(prompt_hub_enabled=True)), \
         patch("app.services.prompt_manager.get_prompt_store", return_value=mock_prompt_store):

        from app.core.app import create_app
        app = create_app()

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/prompt/format/greeting", json={"name": "World"})

    assert resp.status_code == 200
    mock_prompt_store.format_prompt.assert_awaited_once()


def test_chain_hub_get_chains_route():
    """GET /chains invokes chainStore.get_chains."""
    mock_chain_store = MagicMock()
    mock_chain_store.get_chains = AsyncMock(return_value={"summarize": {"model": "test"}})

    with patch("app.core.app.settings", _mock_settings(chain_hub_enabled=True)), \
         patch("app.services.chain_manager.get_chain_store", return_value=mock_chain_store):

        from app.core.app import create_app
        app = create_app()

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/chains")

    assert resp.status_code == 200
    assert "summarize" in resp.json()
    mock_chain_store.get_chains.assert_awaited_once()


def test_chain_hub_execute_route():
    """POST /chain/execute/{name} invokes chainStore.execute."""
    mock_chain_store = MagicMock()
    mock_chain_store.execute = AsyncMock(return_value={"output": "summary text"})

    with patch("app.core.app.settings", _mock_settings(chain_hub_enabled=True)), \
         patch("app.services.chain_manager.get_chain_store", return_value=mock_chain_store):

        from app.core.app import create_app
        app = create_app()

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/chain/execute/summarize", json={"topic": "AI"})

    assert resp.status_code == 200
    mock_chain_store.execute.assert_awaited_once()
    # Verify metadata was constructed with chain_type=chain
    call_args = mock_chain_store.execute.call_args
    metadata = call_args[0][3] if len(call_args[0]) > 3 else call_args[1].get("metadata")
    assert metadata.chain_type.value == "chain"
    assert metadata.chain_name == "summarize"


def test_prompt_execute_route():
    """POST /prompt/execute/{name} requires both prompt_hub and chain_hub."""
    mock_prompt_store = MagicMock()
    mock_prompt_store.format_prompt = AsyncMock(return_value="Tell me about AI")
    mock_prompt_store.get_prompts = AsyncMock(return_value={})

    mock_chain_store = MagicMock()
    mock_chain_store.execute_prompt = AsyncMock(return_value={"output": "AI is..."})
    mock_chain_store.get_chains = AsyncMock(return_value={})

    with patch("app.core.app.settings", _mock_settings(prompt_hub_enabled=True, chain_hub_enabled=True)), \
         patch("app.services.prompt_manager.get_prompt_store", return_value=mock_prompt_store), \
         patch("app.services.chain_manager.get_chain_store", return_value=mock_chain_store):

        from app.core.app import create_app
        app = create_app()

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/prompt/execute/explain", json={"topic": "AI"})

    assert resp.status_code == 200
    mock_prompt_store.format_prompt.assert_awaited_once()
    mock_chain_store.execute_prompt.assert_awaited_once()
    # Verify metadata was constructed with chain_type=prompt
    call_kwargs = mock_chain_store.execute_prompt.call_args[1]
    assert call_kwargs["metadata"].chain_type.value == "prompt"
    assert call_kwargs["metadata"].chain_name == "explain"
