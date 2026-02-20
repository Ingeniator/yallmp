from app.services.token_manager import OIDCTokenManager
from app.core.config import settings
from httpx import AsyncClient
from app.core.logging_config import setup_logging

logger = setup_logging()

token_manager = None
if settings.proxy_authorization_type == "BEARER":
    token_manager = OIDCTokenManager(settings.proxy_oidc_authorization_url, settings.proxy_oidc_credentials)


async def get_authorization_headers(client: AsyncClient):
    custom_headers = {}
    if token_manager:
        token = await token_manager.get_token(client)
        custom_headers["Authorization"] = f"Bearer {token}"
    elif settings.proxy_authorization_type == "APIKEY":
        if not settings.proxy_api_key:
            logger.error("proxy_authorization_type is APIKEY but proxy_api_key is not set")
        else:
            custom_headers["X-API-KEY"] = settings.proxy_api_key
    return custom_headers
