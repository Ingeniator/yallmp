import pytest
from unittest.mock import patch, AsyncMock, MagicMock
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
        mock_settings.llm_hub_enabled = False
        mock_settings.version = "0.0.1-test"

        from app.core.app import create_app
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "ok"
        assert all(v == "ok" for v in data["components"].values())


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
        mock_settings.version = "0.0.1-test"

        from app.core.app import create_app
        app = create_app()

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            assert app.state.client is None


def test_lifespan_with_proxy():
    """Lifespan creates and closes the async client when proxy is enabled."""
    mock_client = AsyncMock()

    with patch("app.core.app.settings") as mock_settings, \
         patch("app.core.proxy.create_async_client", AsyncMock(return_value=mock_client)):
        mock_settings.app_name = "TestApp"
        mock_settings.debug = False
        mock_settings.root_path = ""
        mock_settings.allowed_origins = ["*"]
        mock_settings.proxy_enabled = True
        mock_settings.prompt_hub_enabled = False
        mock_settings.chain_hub_enabled = False
        mock_settings.llm_hub_enabled = False
        mock_settings.version = "0.0.1-test"

        from app.core.app import create_app
        app = create_app()

        with TestClient(app, raise_server_exceptions=False) as client:
            assert app.state.client is mock_client

    mock_client.aclose.assert_awaited_once()


# --- Feature-flag route handler tests ---


def _mock_settings(**overrides):
    """Return a patch context for app.core.app.settings with given feature flags."""
    defaults = dict(
        app_name="TestApp", debug=False, root_path="", allowed_origins=["*"],
        proxy_enabled=False, prompt_hub_enabled=False, chain_hub_enabled=False,
        llm_hub_enabled=False,
        version="0.0.1-test",
    )
    defaults.update(overrides)

    class FakeSettings:
        pass

    s = FakeSettings()
    for k, v in defaults.items():
        setattr(s, k, v)
    return s


def test_proxy_llm_version_route():
    """GET /llm/version invokes get_model_version via the proxy branch."""
    mock_get_model_version = AsyncMock(return_value={"version": "GigaChat-9b-128k-base:3.0"})
    mock_get_auth_headers = AsyncMock(return_value={"Authorization": "Bearer tok"})
    mock_http_client = AsyncMock()

    with patch("app.core.app.settings", _mock_settings(proxy_enabled=True)), \
         patch("app.core.proxy.get_model_version", mock_get_model_version), \
         patch("app.core.proxy.proxy_request_with_retries", AsyncMock()), \
         patch("app.services.llm_authentication.get_authorization_headers", mock_get_auth_headers), \
         patch("app.core.proxy.create_async_client", AsyncMock(return_value=mock_http_client)):

        from app.core.app import create_app
        app = create_app()

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/llm/version")

    assert resp.status_code == 200
    assert resp.json() == {"version": "GigaChat-9b-128k-base:3.0"}
    mock_get_auth_headers.assert_awaited_once()
    mock_get_model_version.assert_awaited_once()


def test_proxy_llm_path_route():
    """POST /llm/{path} invokes proxy_request_with_retries."""
    mock_proxy = AsyncMock(return_value=JSONResponse(content={"choices": []}, status_code=200))
    mock_get_auth_headers = AsyncMock(return_value={})
    mock_http_client = AsyncMock()

    with patch("app.core.app.settings", _mock_settings(proxy_enabled=True)), \
         patch("app.core.proxy.proxy_request_with_retries", mock_proxy), \
         patch("app.core.proxy.get_model_version", AsyncMock()), \
         patch("app.services.llm_authentication.get_authorization_headers", mock_get_auth_headers), \
         patch("app.core.proxy.create_async_client", AsyncMock(return_value=mock_http_client)):

        from app.core.app import create_app
        app = create_app()

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/llm/v1/chat/completions", json={"model": "test"})

    assert resp.status_code == 200
    mock_proxy.assert_awaited_once()


def test_prompt_hub_get_prompts_route():
    """GET /prompts invokes promptStore.get_prompts."""
    mock_prompt_store = MagicMock()
    mock_prompt_store.get_prompts = AsyncMock(return_value={"greeting": {"input_variables": ["name"]}})

    with patch("app.core.app.settings", _mock_settings(prompt_hub_enabled=True)), \
         patch("app.services.prompt_manager.promptStore", mock_prompt_store):

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
         patch("app.services.prompt_manager.promptStore", mock_prompt_store):

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
         patch("app.services.chain_manager.chainStore", mock_chain_store):

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
         patch("app.services.chain_manager.chainStore", mock_chain_store):

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
         patch("app.services.prompt_manager.promptStore", mock_prompt_store), \
         patch("app.services.chain_manager.chainStore", mock_chain_store):

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
