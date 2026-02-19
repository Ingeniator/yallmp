from httpx import AsyncClient
import time
import uuid
from app.core.logging_config import setup_logging
from app.core.config import settings

logger = setup_logging()

class OIDCTokenManager:
    def __init__(self, authorization_url, credentials):
        self.authorization_url = authorization_url
        self.credentials = credentials
        self.token = None
        self.expires_at = 0  # Timestamp when the token expires

    async def fetch_token(self, client: AsyncClient):
        """Fetch a new access token using client credentials grant."""
        response = await client.post(
            self.authorization_url,
            data={
                "scope": settings.chain_default_scope
            },
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "RqUID": f"{uuid.uuid4()}",
                "Authorization": f"Basic {self.credentials}"
            }
        )
        response.raise_for_status()
        data = response.json()
        self.token = data.get("access_token", data.get("tok", ""))
        self.expires_at = data.get("expires_at", data.get("exp", "")) - 20*1000  # Refresh slightly before expiration 20s before
        logger.debug(self.expires_at)
    
    async def get_token(self, client: AsyncClient):
        """Ensure the token is fresh and return it."""
        if not self.token or time.time()* 1000 >= self.expires_at:
            logger.debug("time to refresh token")
            try:
                await self.fetch_token(client)
            except Exception as e:
                logger.error(f"Error fetching token: {type(e).__name__}")
                return "Error fetching token"
        return self.token
