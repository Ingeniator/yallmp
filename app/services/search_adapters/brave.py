from __future__ import annotations

from httpx import AsyncClient

from app.schemas.search import SearchRequest, SearchResponse, SearchResult
from app.services.search_adapters.base import SearchAdapter


class BraveAdapter(SearchAdapter):
    """Adapter for the Brave Search API (https://api.search.brave.com/app/documentation).

    Auth: API key sent as ``X-Subscription-Token`` header.
    Endpoint: GET {base_url}/res/v1/web/search
    """

    async def search(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
        req: SearchRequest,
        provider_name: str,
    ) -> SearchResponse:
        params: dict = {
            "q": req.query,
            "count": req.num_results,
        }

        response = await client.get(
            "/res/v1/web/search",
            params=params,
            headers=auth_headers,
        )
        response.raise_for_status()
        data = response.json()

        web = data.get("web", {})
        raw_results = web.get("results", [])

        results = []
        for idx, r in enumerate(raw_results):
            score = max(0.0, 1.0 - idx * 0.1)
            # 'description' is Brave's snippet field; fall back to extra_snippets
            content = r.get("description") or ""
            if not content and r.get("extra_snippets"):
                content = " ".join(r["extra_snippets"])
            results.append(
                SearchResult(
                    url=r.get("url", ""),
                    title=r.get("title", ""),
                    content=content,
                    score=score,
                    raw_content=None,
                )
            )

        return SearchResponse(
            results=results,
            provider=provider_name,
            query=req.query,
            answer=None,
        )
