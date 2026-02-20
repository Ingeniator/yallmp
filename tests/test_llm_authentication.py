import pytest
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.mark.asyncio
async def test_bearer_auth():
    mock_tm = MagicMock()
    mock_tm.get_token = AsyncMock(return_value="test-token-123")

    with patch("app.services.llm_authentication.token_manager", mock_tm), \
         patch("app.services.llm_authentication.settings") as mock_settings:
        mock_settings.proxy_authorization_type = "BEARER"
        mock_settings.proxy_api_key = None

        from app.services.llm_authentication import get_authorization_headers
        client = AsyncMock()
        headers = await get_authorization_headers(client)

    assert headers["Authorization"] == "Bearer test-token-123"
    mock_tm.get_token.assert_awaited_once_with(client)


@pytest.mark.asyncio
async def test_apikey_auth():
    with patch("app.services.llm_authentication.token_manager", None), \
         patch("app.services.llm_authentication.settings") as mock_settings:
        mock_settings.proxy_authorization_type = "APIKEY"
        mock_settings.proxy_api_key = "my-api-key"

        from app.services.llm_authentication import get_authorization_headers
        client = AsyncMock()
        headers = await get_authorization_headers(client)

    assert headers["X-API-KEY"] == "my-api-key"


@pytest.mark.asyncio
async def test_no_auth():
    with patch("app.services.llm_authentication.token_manager", None), \
         patch("app.services.llm_authentication.settings") as mock_settings:
        mock_settings.proxy_authorization_type = "CERT"
        mock_settings.proxy_api_key = None

        from app.services.llm_authentication import get_authorization_headers
        client = AsyncMock()
        headers = await get_authorization_headers(client)

    assert headers == {}
