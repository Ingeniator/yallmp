import httpx
import time

class OIDCTokenManager:
    def __init__(self, authorization_url, client_id, client_secret):
        self.authorization_url = authorization_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.token = None
        self.expires_at = 0  # Timestamp when the token expires

    async def fetch_token(self):
        """Fetch a new access token using client credentials grant."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.authorization_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            response.raise_for_status()
            data = response.json()
            self.token = data["access_token"]
            self.expires_at = time.time() + data["expires_in"] - 10  # Refresh slightly before expiration
    
    async def get_token(self):
        """Ensure the token is fresh and return it."""
        if not self.token or time.time() >= self.expires_at:
            await self.fetch_token()
        return self.token