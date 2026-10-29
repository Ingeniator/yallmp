from __future__ import annotations

import xml.etree.ElementTree as ET

from httpx import AsyncClient

from app.schemas.search import SearchRequest, SearchResponse, SearchResult
from app.services.search_adapters.base import SearchAdapter

# Yandex requires an explicit search type; RU is the general-purpose web index.
_DEFAULT_SEARCH_TYPE = "SEARCH_TYPE_RU"


def _text(element: ET.Element | None) -> str:
    """Flatten an element's mixed content (``<hlword>`` markup) into plain text."""
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def _doc_content(doc: ET.Element) -> str:
    """Snippet for a document: passages, falling back to the headline."""
    passages = doc.find("passages")
    if passages is not None:
        joined = " ".join(
            text for p in passages.findall("passage") if (text := _text(p))
        )
        if joined:
            return joined
    return _text(doc.find("headline"))


class YandexAdapter(SearchAdapter):
    """Adapter for Yandex web search exposed by the gateway.

    Auth: API key sent as ``X-API-Key`` header.
    Endpoint: POST {base_url}/yandex/web-search

    The upstream returns the native Yandex Search API envelope: a JSON object
    whose ``raw_text`` field holds the XML report.
    """

    async def search(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
        req: SearchRequest,
        provider_name: str,
    ) -> SearchResponse:
        # Yandex has no include/exclude domain parameters, so the corresponding
        # SearchRequest fields are not applicable here.
        payload: dict = {
            "query": {
                "searchType": _DEFAULT_SEARCH_TYPE,
                "queryText": req.query,
            },
            "groupSpec": {"groupsOnPage": str(req.num_results)},
        }

        response = await client.post(
            "/yandex/web-search",
            json=payload,
            headers=auth_headers,
        )
        response.raise_for_status()
        data = response.json()

        raw_text = data.get("raw_text") or ""
        if not raw_text:
            return SearchResponse(
                results=[], provider=provider_name, query=req.query, answer=None
            )

        # ElementTree expands internal entities, so a hostile DOCTYPE could
        # exhaust memory. Yandex never sends one — refuse instead of parsing.
        if "<!DOCTYPE" in raw_text:
            raise ValueError("Yandex response contains a DOCTYPE declaration")

        root = ET.fromstring(raw_text)

        results = []
        for idx, doc in enumerate(root.iter("doc")):
            # Yandex exposes no per-document score; derive one from rank as the
            # Exa and Brave adapters do.
            extended = doc.findtext("properties/extended-text")
            results.append(
                SearchResult(
                    url=doc.findtext("url", ""),
                    title=_text(doc.find("title")),
                    content=_doc_content(doc),
                    score=max(0.0, 1.0 - idx * 0.1),
                    raw_content=extended if req.include_raw_content else None,
                )
            )

        return SearchResponse(
            results=results,
            provider=provider_name,
            query=req.query,
            answer=None,
        )
