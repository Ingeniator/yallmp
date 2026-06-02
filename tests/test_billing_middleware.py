"""Tests for BillingMiddleware pre-request limit enforcement."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from app.middlewares.billing_middleware import BillingMiddleware

_LIMITS = {
    "tiers": {
        "tier1": {
            "period": "month",
            "group_limit": 100.0,
            "user_limit": 10.0,
            "alert_threshold": 0.8,
        },
    },
    "orgs": {"acme": "tier1", "default": "tier1"},
}


def _make_app(redis_mock, billing_enabled: bool = True):
    async def llm_endpoint(request: Request):
        return JSONResponse({"ok": True})

    async def other_endpoint(request: Request):
        return JSONResponse({"other": True})

    app = Starlette(routes=[
        Route("/ai/llm/chat", llm_endpoint, methods=["POST"]),
        Route("/ai/other", other_endpoint),
    ])
    app.state.billing_redis = redis_mock
    app.state.billing_limits = _LIMITS
    with patch("app.middlewares.billing_middleware.settings") as mock_settings:
        mock_settings.billing_enabled = billing_enabled
        mock_settings.root_path = "/ai"
        app.add_middleware(BillingMiddleware)
        return app


@pytest.fixture
def allowed_redis():
    r = AsyncMock()
    r.get = AsyncMock(return_value=b"5.0")  # well below limits
    return r


@pytest.fixture
def group_over_limit_redis():
    r = AsyncMock()
    r.get = AsyncMock(return_value=b"100.0")  # at group_limit
    return r


@pytest.fixture
def user_over_limit_redis():
    call_count = 0

    async def _get(key):
        nonlocal call_count
        call_count += 1
        if "group" in key:
            return b"5.0"   # group ok
        return b"10.0"       # user at limit

    r = AsyncMock()
    r.get = _get
    return r


@pytest.fixture
def approaching_threshold_redis():
    r = AsyncMock()
    r.get = AsyncMock(return_value=b"85.0")  # > 80% of 100
    return r


def test_allowed_request_passes(allowed_redis):
    with patch("app.middlewares.billing_middleware.settings") as mock_settings:
        mock_settings.billing_enabled = True
        mock_settings.root_path = "/ai"

        async def llm_endpoint(request: Request):
            return JSONResponse({"ok": True})

        app = Starlette(routes=[Route("/ai/llm/chat", llm_endpoint, methods=["POST"])])
        app.state.billing_redis = allowed_redis
        app.state.billing_limits = _LIMITS
        app.add_middleware(BillingMiddleware)

        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post(
            "/ai/llm/chat",
            headers={"x-group-id": "acme/alice"},
            json={},
        )
        assert resp.status_code == 200


def test_group_limit_reached_returns_429(group_over_limit_redis):
    with patch("app.middlewares.billing_middleware.settings") as mock_settings:
        mock_settings.billing_enabled = True
        mock_settings.root_path = "/ai"

        async def llm_endpoint(request: Request):
            return JSONResponse({"ok": True})

        app = Starlette(routes=[Route("/ai/llm/chat", llm_endpoint, methods=["POST"])])
        app.state.billing_redis = group_over_limit_redis
        app.state.billing_limits = _LIMITS
        app.add_middleware(BillingMiddleware)

        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post(
            "/ai/llm/chat",
            headers={"x-group-id": "acme/alice"},
            json={},
        )
        assert resp.status_code == 429
        assert "group spend limit reached" in resp.json()["error"]


def test_user_limit_reached_returns_429(user_over_limit_redis):
    with patch("app.middlewares.billing_middleware.settings") as mock_settings:
        mock_settings.billing_enabled = True
        mock_settings.root_path = "/ai"

        async def llm_endpoint(request: Request):
            return JSONResponse({"ok": True})

        app = Starlette(routes=[Route("/ai/llm/chat", llm_endpoint, methods=["POST"])])
        app.state.billing_redis = user_over_limit_redis
        app.state.billing_limits = _LIMITS
        app.add_middleware(BillingMiddleware)

        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post(
            "/ai/llm/chat",
            headers={"x-group-id": "acme/alice"},
            json={},
        )
        assert resp.status_code == 429
        assert "user spend limit reached" in resp.json()["error"]


def test_warning_header_added_when_approaching_threshold(approaching_threshold_redis):
    # Use a plain org group_id (no "/") so only group check runs.
    # "unknown-org" falls back to default tier1: limit=100, threshold=0.8 → warns at 85.
    with patch("app.middlewares.billing_middleware.settings") as mock_settings:
        mock_settings.billing_enabled = True
        mock_settings.root_path = "/ai"

        async def llm_endpoint(request: Request):
            return JSONResponse({"ok": True})

        app = Starlette(routes=[Route("/ai/llm/chat", llm_endpoint, methods=["POST"])])
        app.state.billing_redis = approaching_threshold_redis
        app.state.billing_limits = _LIMITS
        app.add_middleware(BillingMiddleware)

        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post(
            "/ai/llm/chat",
            headers={"x-group-id": "unknown-org"},
            json={},
        )
        assert resp.status_code == 200
        assert resp.headers.get("x-billing-warning") == "approaching group limit"


def test_non_llm_path_bypasses_middleware(allowed_redis):
    with patch("app.middlewares.billing_middleware.settings") as mock_settings:
        mock_settings.billing_enabled = True
        mock_settings.root_path = "/ai"

        async def other_endpoint(request: Request):
            return JSONResponse({"other": True})

        app = Starlette(routes=[Route("/ai/other", other_endpoint)])
        app.state.billing_redis = allowed_redis
        app.state.billing_limits = _LIMITS
        app.add_middleware(BillingMiddleware)

        client = TestClient(app)
        resp = client.get("/ai/other")
        assert resp.status_code == 200
        allowed_redis.get.assert_not_called()


def test_billing_disabled_bypasses_checks():
    redis = AsyncMock()

    with patch("app.middlewares.billing_middleware.settings") as mock_settings:
        mock_settings.billing_enabled = False
        mock_settings.root_path = "/ai"

        async def llm_endpoint(request: Request):
            return JSONResponse({"ok": True})

        app = Starlette(routes=[Route("/ai/llm/chat", llm_endpoint, methods=["POST"])])
        app.state.billing_redis = redis
        app.state.billing_limits = _LIMITS
        app.add_middleware(BillingMiddleware)

        client = TestClient(app)
        resp = client.post("/ai/llm/chat", headers={"x-group-id": "acme/alice"}, json={})
        assert resp.status_code == 200
        redis.get.assert_not_called()


def test_redis_error_fails_open():
    redis = AsyncMock()
    redis.get.side_effect = ConnectionError("redis down")

    with patch("app.middlewares.billing_middleware.settings") as mock_settings:
        mock_settings.billing_enabled = True
        mock_settings.root_path = "/ai"

        async def llm_endpoint(request: Request):
            return JSONResponse({"ok": True})

        app = Starlette(routes=[Route("/ai/llm/chat", llm_endpoint, methods=["POST"])])
        app.state.billing_redis = redis
        app.state.billing_limits = _LIMITS
        app.add_middleware(BillingMiddleware)

        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post("/ai/llm/chat", headers={"x-group-id": "acme/alice"}, json={})
        assert resp.status_code == 200
