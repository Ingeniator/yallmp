from __future__ import annotations

from httpx import AsyncClient

from app.schemas.search import SearchRequest, SearchResponse, SearchResult
from app.services.search_adapters.base import SearchAdapter


class TavilyAdapter(SearchAdapter):
    """Adapter for the Tavily Search API (https://docs.tavily.com/docs/rest-api).

    Auth: API key sent as ``X-Tvly-Api-Key`` header.
    Endpoint: POST {base_url}/search
    """

    async def search(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
        req: SearchRequest,
        provider_name: str,
    ) -> SearchResponse:
        payload: dict = {
            "query": req.query,
            "search_depth": req.search_depth,
            "max_results": req.num_results,
            "include_raw_content": req.include_raw_content,
        }
        if req.include_domains:
            payload["include_domains"] = req.include_domains
        if req.exclude_domains:
            payload["exclude_domains"] = req.exclude_domains

        response = await client.post(
            "/search",
            json=payload,
            headers=auth_headers,
        )
        response.raise_for_status()
        data = response.json()

        results = [
            SearchResult(
                url=r.get("url", ""),
                title=r.get("title", ""),
                content=r.get("content", ""),
                score=r.get("score"),
                raw_content=r.get("raw_content"),
            )
            for r in data.get("results", [])
        ]

        return SearchResponse(
            results=results,
            provider=provider_name,
            query=req.query,
            answer=data.get("answer"),
        )
