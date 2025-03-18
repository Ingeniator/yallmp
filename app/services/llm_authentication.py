from app.services.token_manager import OIDCTokenManager
from app.core.config import settings

token_manager = None
if settings.proxy_authorization_type == "BEARER":
    token_manager = OIDCTokenManager(settings.proxy_oidc_authorization_url, settings.proxy_oidc_client_id, settings.proxy_oidc_client_secret)

async def get_authorization_headers():
    custom_headers = {}
    if token_manager:
        token = await token_manager.get_token()
        custom_headers["Authorization"] = f"Bearer {token}"
    elif settings.proxy_authorization_type == "APIKEY":
        custom_headers["Authorization"] = f"Bearer {settings.proxy_api_key}"
    return custom_headers
