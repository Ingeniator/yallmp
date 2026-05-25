from __future__ import annotations

from httpx import AsyncClient

from app.schemas.search import SearchRequest, SearchResponse, SearchResult
from app.services.search_adapters.base import SearchAdapter

# Max chars of text content to request per result; keeps response sizes sane.
_DEFAULT_MAX_CHARS = 1000


class ExaAdapter(SearchAdapter):
    """Adapter for the Exa Search API (https://docs.exa.ai/reference/search).

    Auth: API key sent as ``x-api-key`` header.
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
            "numResults": req.num_results,
            "contents": {
                "text": {
                    "maxCharacters": _DEFAULT_MAX_CHARS
                    if not req.include_raw_content
                    else None,
                },
            },
        }
        if req.include_domains:
            payload["includeDomains"] = req.include_domains
        if req.exclude_domains:
            payload["excludeDomains"] = req.exclude_domains

        response = await client.post(
            "/search",
            json=payload,
            headers=auth_headers,
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for idx, r in enumerate(data.get("results", [])):
            # Exa returns autopromptString-ranked results; derive a score from
            # position (1.0 for rank 1, diminishing) when a score is absent.
            score = r.get("score") or max(0.0, 1.0 - idx * 0.1)
            content = r.get("text") or r.get("summary") or ""
            raw = r.get("text") if req.include_raw_content else None
            results.append(
                SearchResult(
                    url=r.get("url", ""),
                    title=r.get("title", ""),
                    content=content,
                    score=score,
                    raw_content=raw,
                )
            )

        return SearchResponse(
            results=results,
            provider=provider_name,
            query=req.query,
            answer=None,  # Exa doesn't generate a synthesized answer
        )
