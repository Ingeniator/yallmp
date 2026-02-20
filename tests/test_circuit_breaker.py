import pytest
import time
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import Response as HTTPXResponse, Request as HTTPXRequest, ConnectError, RequestError
from fastapi.responses import JSONResponse

import app.core.proxy as proxy_mod
from app.core.proxy import CircuitBreaker


def _ok_response():
    return HTTPXResponse(
        status_code=200,
        json={"ok": True},
        request=HTTPXRequest("GET", "http://test"),
    )


def _error_response(status_code):
    return HTTPXResponse(
        status_code=status_code,
        json={"error": "fail"},
        request=HTTPXRequest("GET", "http://test"),
    )


@pytest.fixture(autouse=True)
def _reset_circuit():
    """Reset circuit breaker state before each test."""
    proxy_mod.circuit_breaker = CircuitBreaker()


@pytest.mark.asyncio
async def test_successful_request_clears_failures():
    proxy_mod.circuit_breaker.failure_timestamps.append(1.0)
    func = AsyncMock(return_value=_ok_response())

    with patch.object(proxy_mod, "settings") as s:
        s.proxy_max_retries = 0
        s.proxy_base_delay = 0
        s.proxy_backoff_factor = 1
        s.proxy_failure_threshold = 0
        s.proxy_window_size = 60
        s.proxy_recovery_time = 30

        result = await proxy_mod.exponential_backoff_retry(func)

    assert isinstance(result, HTTPXResponse)
    assert result.status_code == 200
    assert proxy_mod.circuit_breaker.failure_timestamps == []


@pytest.mark.asyncio
async def test_429_triggers_retry():
    rate_limited = _error_response(429)
    ok = _ok_response()
    func = AsyncMock(side_effect=[rate_limited, ok])

    with patch.object(proxy_mod, "settings") as s:
        s.proxy_max_retries = 1
        s.proxy_base_delay = 0
        s.proxy_backoff_factor = 1
        s.proxy_failure_threshold = 0
        s.proxy_window_size = 60
        s.proxy_recovery_time = 30

        result = await proxy_mod.exponential_backoff_retry(func)

    assert isinstance(result, HTTPXResponse)
    assert result.status_code == 200
    assert func.call_count == 2


@pytest.mark.asyncio
async def test_500_triggers_retry():
    err = _error_response(500)
    ok = _ok_response()
    func = AsyncMock(side_effect=[err, ok])

    with patch.object(proxy_mod, "settings") as s:
        s.proxy_max_retries = 1
        s.proxy_base_delay = 0
        s.proxy_backoff_factor = 1
        s.proxy_failure_threshold = 0
        s.proxy_window_size = 60
        s.proxy_recovery_time = 30

        result = await proxy_mod.exponential_backoff_retry(func)

    assert isinstance(result, HTTPXResponse)
    assert result.status_code == 200


@pytest.mark.asyncio
async def test_circuit_breaker_activates():
    func = AsyncMock(return_value=_error_response(500))

    with patch.object(proxy_mod, "settings") as s:
        s.proxy_max_retries = 0
        s.proxy_base_delay = 0
        s.proxy_backoff_factor = 1
        s.proxy_failure_threshold = 1
        s.proxy_window_size = 60
        s.proxy_recovery_time = 30

        result = await proxy_mod.exponential_backoff_retry(func)

    assert isinstance(result, JSONResponse)
    assert result.status_code == 503
    assert proxy_mod.circuit_breaker.is_open is True


@pytest.mark.asyncio
async def test_circuit_breaker_returns_503_while_open():
    proxy_mod.circuit_breaker.is_open = True
    proxy_mod.circuit_breaker.open_time = time.time()
    func = AsyncMock()

    with patch.object(proxy_mod, "settings") as s:
        s.proxy_recovery_time = 9999

        result = await proxy_mod.exponential_backoff_retry(func)

    assert isinstance(result, JSONResponse)
    assert result.status_code == 503
    func.assert_not_called()


@pytest.mark.asyncio
async def test_connect_error_handled():
    req = HTTPXRequest("GET", "http://test")
    func = AsyncMock(side_effect=ConnectError("connection refused", request=req))

    with patch.object(proxy_mod, "settings") as s:
        s.proxy_max_retries = 0
        s.proxy_base_delay = 0
        s.proxy_backoff_factor = 1
        s.proxy_failure_threshold = 0
        s.proxy_window_size = 60
        s.proxy_recovery_time = 30

        result = await proxy_mod.exponential_backoff_retry(func)

    assert isinstance(result, JSONResponse)
    assert result.status_code == 523


@pytest.mark.asyncio
async def test_request_error_handled():
    req = HTTPXRequest("GET", "http://test")
    func = AsyncMock(side_effect=RequestError("read timeout", request=req))

    with patch.object(proxy_mod, "settings") as s:
        s.proxy_max_retries = 0
        s.proxy_base_delay = 0
        s.proxy_backoff_factor = 1
        s.proxy_failure_threshold = 0
        s.proxy_window_size = 60
        s.proxy_recovery_time = 30

        result = await proxy_mod.exponential_backoff_retry(func)

    assert isinstance(result, JSONResponse)
    assert result.status_code == 523
