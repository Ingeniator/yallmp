from unittest.mock import patch
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


import pytest  # noqa: E402

from app.middlewares.metrics_middleware import _normalize_path  # noqa: E402


@pytest.mark.parametrize("path,expected", [
    ("/ai/llm/v1/files/file-abc123XYZ789", "/ai/llm/v1/files/:id"),
    ("/ai/llm/v1/files/file-abc123XYZ789/content", "/ai/llm/v1/files/:id/content"),
    ("/ai/llm/v1/fine_tuning/jobs/ftjob-abc123def", "/ai/llm/v1/fine_tuning/jobs/:id"),
    ("/ai/llm/@openai/v1/batches/batch_67a1b2c3d4", "/ai/llm/@openai/v1/batches/:id"),
    ("/ai/llm/v1/models/ft:gpt-4o-mini:org::9xYz", "/ai/llm/v1/models/:id"),
    ("/ai/llm/v1/items/123", "/ai/llm/v1/items/:id"),
    ("/ai/llm/v1/x/550e8400-e29b-41d4-a716-446655440000", "/ai/llm/v1/x/:id"),
])
def test_normalize_path_replaces_ids(path, expected):
    assert _normalize_path(path) == expected


@pytest.mark.parametrize("path", [
    "/ai/llm/v1/chat/completions",
    "/ai/llm/v1/fine_tuning/jobs",
    "/ai/llm/@openai/v1/files",
    "/ai/llm/v1/models/gpt-4o",
    "/ai/llm/v1/models",
])
def test_normalize_path_keeps_api_segments(path):
    assert _normalize_path(path) == path
