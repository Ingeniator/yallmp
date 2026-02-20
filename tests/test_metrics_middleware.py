import pytest
from unittest.mock import patch, MagicMock
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from app.middlewares.metrics_middleware import PrometheusMiddleware


def _make_app():
    async def homepage(request):
        return JSONResponse({"ok": True})

    async def upload(request):
        return JSONResponse({"uploaded": True})

    app = Starlette(routes=[
        Route("/", homepage),
        Route("/upload", upload, methods=["POST"]),
    ])
    app.add_middleware(PrometheusMiddleware)
    return app


@patch("app.middlewares.metrics_middleware.REQUEST_DURATION")
@patch("app.middlewares.metrics_middleware.REQUEST_COUNT")
def test_normal_request_increments_metrics(mock_count, mock_duration):
    client = TestClient(_make_app())
    resp = client.get("/")
    assert resp.status_code == 200

    mock_count.labels.assert_called()
    mock_count.labels.return_value.inc.assert_called()
    mock_duration.labels.assert_called()
    mock_duration.labels.return_value.observe.assert_called()


@patch("app.middlewares.metrics_middleware.REQUEST_DURATION")
@patch("app.middlewares.metrics_middleware.REQUEST_COUNT")
def test_stream_request_skips_metrics(mock_count, mock_duration):
    client = TestClient(_make_app())
    resp = client.post(
        "/upload",
        files={"file": ("test.txt", b"content", "text/plain")},
    )
    assert resp.status_code == 200
    mock_count.labels.assert_not_called()
    mock_duration.labels.assert_not_called()
