import pytest
from unittest.mock import AsyncMock
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from app.middlewares.logging_middleware import LoggingMiddleware


def _make_app():
    async def homepage(request):
        return JSONResponse({"ok": True})

    async def upload(request):
        return JSONResponse({"uploaded": True})

    app = Starlette(routes=[
        Route("/", homepage),
        Route("/upload", upload, methods=["POST"]),
    ])
    app.add_middleware(LoggingMiddleware)
    return app


def test_normal_get_passes_through():
    client = TestClient(_make_app())
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_multipart_post_skipped():
    client = TestClient(_make_app())
    resp = client.post(
        "/upload",
        files={"file": ("test.txt", b"content", "text/plain")},
    )
    assert resp.status_code == 200
    assert resp.json() == {"uploaded": True}


def test_chunked_post_skipped():
    client = TestClient(_make_app())
    resp = client.post(
        "/upload",
        content=b"some data",
        headers={"transfer-encoding": "chunked", "content-type": "application/json"},
    )
    assert resp.status_code == 200
