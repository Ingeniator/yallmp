from __future__ import annotations

import uuid

from httpx import AsyncClient

from app.schemas.search import SearchRequest, SearchResponse, SearchResult
from app.services.search_adapters.base import SearchAdapter


class GigaSearchAdapter(SearchAdapter):
    """Adapter for the GigaSearch v3 API.

    Auth: API key sent via gateway token; provider-specific headers
    (``X-Client-Id``, ``X-Surface-Id``, ``X-Request-Id``) are injected from
    *auth_headers* returned by the corresponding ``SearchProvider``.

    Endpoint: POST {base_url}/v3/search
    """

    async def search(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
        req: SearchRequest,
        provider_name: str,
    ) -> SearchResponse:
        # -- build request payload --
        payload: dict = {
            "query": req.query,
            "limit": req.num_results,
            "search_mode": "advanced",
        }

        # -- required GigaSearch headers --
        headers = {
            **auth_headers,
            "X-Request-Id": auth_headers.get("X-Request-Id", str(uuid.uuid4())),
        }

        response = await client.post(
            "/v3/search",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()

        # -- check search_metadata for upstream errors --
        metadata = data.get("search_metadata", {})
        if metadata.get("status") == "error":
            # Surface the error description to the caller via an empty
            # result set rather than raising, so the orchestrator can
            # decide how to handle partial failures.
            return SearchResponse(
                results=[],
                provider=provider_name,
                query=req.query,
                answer=None,
            )

        # -- map documents → SearchResult --
        documents = data.get("documents") or []
        results = []
        for idx, doc in enumerate(documents):
            # GigaSearch does not return a relevance score —
            # derive one from position (1.0 for rank 1, diminishing).
            score = max(0.0, 1.0 - idx * 0.1)

            results.append(
                SearchResult(
                    url=doc.get("url") or "",
                    title=doc.get("title", ""),
                    content=doc.get("content", ""),
                    score=score,
                    raw_content=None,
                )
            )

        return SearchResponse(
            results=results,
            provider=provider_name,
            query=req.query,
            answer=None,  # GigaSearch does not generate a synthesised answer
        )
