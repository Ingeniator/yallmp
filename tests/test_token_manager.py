import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.token_manager import OIDCTokenManager


@pytest.fixture
def manager():
    return OIDCTokenManager("http://auth.example.com/token", "base64creds")


@pytest.mark.asyncio
async def test_fetch_token_stores_token(manager):
    now_ms = int(time.time() * 1000) + 3600_000
    mock_response = MagicMock()
    mock_response.json.return_value = {"access_token": "tok-abc", "expires_at": now_ms}
    mock_response.raise_for_status = MagicMock()

    client = AsyncMock()
    client.post = AsyncMock(return_value=mock_response)

    await manager.fetch_token(client)
    assert manager.token == "tok-abc"
    assert manager.expires_at == now_ms - 20_000


@pytest.mark.asyncio
async def test_get_token_returns_cached(manager):
    manager.token = "cached-token"
    manager.expires_at = (time.time() + 3600) * 1000  # far future in ms

    client = AsyncMock()
    token = await manager.get_token(client)
    assert token == "cached-token"
    client.post.assert_not_called()


@pytest.mark.asyncio
async def test_get_token_refreshes_when_expired(manager):
    manager.token = "old-token"
    manager.expires_at = 0  # expired

    now_ms = int(time.time() * 1000) + 3600_000
    mock_response = MagicMock()
    mock_response.json.return_value = {"access_token": "new-token", "expires_at": now_ms}
    mock_response.raise_for_status = MagicMock()

    client = AsyncMock()
    client.post = AsyncMock(return_value=mock_response)

    token = await manager.get_token(client)
    assert token == "new-token"
    client.post.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_token_handles_fetch_error(manager):
    manager.token = None
    manager.expires_at = 0

    client = AsyncMock()
    client.post = AsyncMock(side_effect=Exception("network error"))

    with pytest.raises(Exception, match="network error"):
        await manager.get_token(client)
