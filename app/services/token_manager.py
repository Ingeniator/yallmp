from httpx import AsyncClient
import time
import uuid
from app.core.logging_config import setup_logging
from app.core.config import settings

logger = setup_logging()


class OIDCTokenManager:
    def __init__(self, authorization_url, credentials, scope=None):
        self.authorization_url = authorization_url
        self.credentials = credentials
        self.scope = scope
        self.token = None
        self.expires_at = 0  # Timestamp in milliseconds when the token expires

    async def fetch_token(self, client: AsyncClient):
        """Fetch a new access token using client credentials grant."""
        response = await client.post(
            self.authorization_url,
            data={
                "scope": self.scope or settings.chain_default_scope
            },
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "RqUID": f"{uuid.uuid4()}",
                "Authorization": f"Basic {self.credentials}"
            }
        )
        response.raise_for_status()
        data = response.json()
        self.token = data.get("access_token") or data.get("tok")
        if not self.token:
            raise ValueError("Token response missing 'access_token' and 'tok' fields")

        # expires_at / exp are expected in milliseconds
        expires_at = data.get("expires_at") or data.get("exp")
        if expires_at is None:
            raise ValueError("Token response missing 'expires_at' and 'exp' fields")
        self.expires_at = int(expires_at) - 20_000  # Refresh 20s before expiration
        logger.debug(f"Token expires_at (ms): {self.expires_at}")

    async def get_token(self, client: AsyncClient):
        """Ensure the token is fresh and return it."""
        now_ms = int(time.time() * 1000)
        if not self.token or now_ms >= self.expires_at:
            logger.debug("time to refresh token")
            await self.fetch_token(client)
        return self.token
