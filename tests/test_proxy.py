import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import AsyncClient, Response as HTTPXResponse, Request as HTTPXRequest
from starlette.testclient import TestClient
from starlette.requests import Request
from starlette.datastructures import URL


@pytest.mark.asyncio
async def test_proxy_passes_query_parameters():
    """Query parameters from the original request must be forwarded to the upstream."""
    from app.core.proxy import proxy_request_with_retries

    # Build a fake Starlette request with query params
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/llm/v1/models",
        "query_string": b"limit=10&offset=0",
        "headers": [
            (b"host", b"localhost"),
            (b"accept", b"application/json"),
        ],
        "root_path": "",
    }
    request = Request(scope, receive=AsyncMock(return_value={"type": "http.request", "body": b""}))

    # Mock the httpx AsyncClient
    mock_response = HTTPXResponse(
        status_code=200,
        json={"data": []},
        request=HTTPXRequest("GET", "http://upstream/v1/models?limit=10&offset=0"),
    )
    mock_client = AsyncMock(spec=AsyncClient)
    mock_client.request = AsyncMock(return_value=mock_response)

    with patch("app.core.proxy.settings") as mock_settings:
        mock_settings.proxy_target_url = "http://upstream"
        mock_settings.proxy_exclude_headers = "host,authorization"
        mock_settings.proxy_max_retries = 0
        mock_settings.proxy_base_delay = 0.1
        mock_settings.proxy_backoff_factor = 2.0
        mock_settings.proxy_failure_threshold = 0
        mock_settings.proxy_window_size = 60
        mock_settings.proxy_recovery_time = 30

        response = await proxy_request_with_retries(
            client=mock_client,
            path="v1/models",
            request=request,
            custom_headers={},
        )

    # Verify the upstream URL included query parameters
    call_args = mock_client.request.call_args
    called_url = call_args[0][1]  # positional: method, url
    assert "?" in called_url, f"Query parameters missing from upstream URL: {called_url}"
    assert "limit=10" in called_url
    assert "offset=0" in called_url


@pytest.mark.asyncio
async def test_proxy_works_without_query_parameters():
    """Requests without query params should not append a '?' to the URL."""
    from app.core.proxy import proxy_request_with_retries

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/llm/v1/models",
        "query_string": b"",
        "headers": [
            (b"host", b"localhost"),
            (b"accept", b"application/json"),
        ],
        "root_path": "",
    }
    request = Request(scope, receive=AsyncMock(return_value={"type": "http.request", "body": b""}))

    mock_response = HTTPXResponse(
        status_code=200,
        json={"data": []},
        request=HTTPXRequest("GET", "http://upstream/v1/models"),
    )
    mock_client = AsyncMock(spec=AsyncClient)
    mock_client.request = AsyncMock(return_value=mock_response)

    with patch("app.core.proxy.settings") as mock_settings:
        mock_settings.proxy_target_url = "http://upstream"
        mock_settings.proxy_exclude_headers = "host,authorization"
        mock_settings.proxy_max_retries = 0
        mock_settings.proxy_base_delay = 0.1
        mock_settings.proxy_backoff_factor = 2.0
        mock_settings.proxy_failure_threshold = 0
        mock_settings.proxy_window_size = 60
        mock_settings.proxy_recovery_time = 30

        response = await proxy_request_with_retries(
            client=mock_client,
            path="v1/models",
            request=request,
            custom_headers={},
        )

    called_url = mock_client.request.call_args[0][1]
    assert called_url == "http://upstream/v1/models", f"Unexpected URL: {called_url}"
    assert "?" not in called_url
