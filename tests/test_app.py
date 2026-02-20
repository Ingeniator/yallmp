import pytest
from unittest.mock import patch
from starlette.testclient import TestClient


def test_health_all_disabled():
    with patch("app.core.app.settings") as mock_settings:
        mock_settings.app_name = "TestApp"
        mock_settings.debug = False
        mock_settings.root_path = ""
        mock_settings.allowed_origins = ["*"]
        mock_settings.proxy_enabled = False
        mock_settings.prompt_hub_enabled = False
        mock_settings.chain_hub_enabled = False
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
        mock_settings.version = "0.0.1-test"

        from app.core.app import create_app
        app = create_app()

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            assert app.state.client is None


def test_lifespan_with_proxy():
    """Lifespan creates and closes the async client when proxy is enabled."""
    from unittest.mock import AsyncMock

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
        mock_settings.version = "0.0.1-test"

        from app.core.app import create_app
        app = create_app()

        with TestClient(app, raise_server_exceptions=False) as client:
            assert app.state.client is mock_client

    mock_client.aclose.assert_awaited_once()
