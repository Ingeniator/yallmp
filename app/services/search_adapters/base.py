from __future__ import annotations

from abc import ABC, abstractmethod

from httpx import AsyncClient

from app.schemas.search import SearchRequest, SearchResponse


class SearchAdapter(ABC):
    """Translate a normalized SearchRequest into provider-specific HTTP calls
    and map the response back to a normalized SearchResponse."""

    @abstractmethod
    async def search(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
        req: SearchRequest,
        provider_name: str,
    ) -> SearchResponse:
        """Execute the search and return a normalized response.

        Args:
            client: Shared AsyncClient configured for this provider.
            auth_headers: Auth headers returned by SearchProvider.get_auth_headers().
            req: The normalized search request.
            provider_name: The provider's name string (for the response).

        Raises:
            httpx.HTTPStatusError: On upstream 4xx/5xx responses.
        """
