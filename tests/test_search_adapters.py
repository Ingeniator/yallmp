"""Tests for search provider adapters.

All HTTP calls are mocked — no real network traffic.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.schemas.search import SearchRequest, SearchResponse
from app.services.search_adapters.tavily import TavilyAdapter
from app.services.search_adapters.exa import ExaAdapter
from app.services.search_adapters.brave import BraveAdapter
from app.services.search_adapters import get_adapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(json_data: dict, status_code: int = 200):
    """Return an AsyncMock httpx.Response-like object."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json = MagicMock(return_value=json_data)
    resp.raise_for_status = MagicMock()  # no-op on success
    return resp


def _mock_client(response):
    """Return a mock AsyncClient whose .post/.get return the given response."""
    client = AsyncMock()
    client.post = AsyncMock(return_value=response)
    client.get = AsyncMock(return_value=response)
    return client


# ---------------------------------------------------------------------------
# TavilyAdapter
# ---------------------------------------------------------------------------

TAVILY_RESPONSE = {
    "query": "python asyncio",
    "answer": "Asyncio is a library to write concurrent code.",
    "results": [
        {
            "url": "https://docs.python.org/asyncio",
            "title": "asyncio — Python docs",
            "content": "asyncio is a library...",
            "score": 0.95,
            "raw_content": None,
        },
        {
            "url": "https://realpython.com/async-io-python/",
            "title": "Async IO in Python",
            "content": "A walkthrough of async IO...",
            "score": 0.87,
        },
    ],
}


@pytest.mark.asyncio
async def test_tavily_basic_search():
    adapter = TavilyAdapter()
    client = _mock_client(_mock_response(TAVILY_RESPONSE))
    req = SearchRequest(query="python asyncio", num_results=5)

    result = await adapter.search(client, {"X-Tvly-Api-Key": "key"}, req, "tavily")

    assert isinstance(result, SearchResponse)
    assert result.provider == "tavily"
    assert result.query == "python asyncio"
    assert result.answer == "Asyncio is a library to write concurrent code."
    assert len(result.results) == 2
    assert result.results[0].url == "https://docs.python.org/asyncio"
    assert result.results[0].score == 0.95


@pytest.mark.asyncio
async def test_tavily_passes_correct_payload():
    adapter = TavilyAdapter()
    resp = _mock_response(TAVILY_RESPONSE)
    client = _mock_client(resp)
    req = SearchRequest(
        query="test query",
        num_results=3,
        search_depth="advanced",
        include_domains=["example.com"],
        exclude_domains=["spam.com"],
    )

    await adapter.search(client, {}, req, "tavily")

    call_kwargs = client.post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs.args[1]
    assert payload["query"] == "test query"
    assert payload["max_results"] == 3
    assert payload["search_depth"] == "advanced"
    assert payload["include_domains"] == ["example.com"]
    assert payload["exclude_domains"] == ["spam.com"]


@pytest.mark.asyncio
async def test_tavily_empty_results():
    adapter = TavilyAdapter()
    client = _mock_client(_mock_response({"query": "nothing", "results": []}))
    req = SearchRequest(query="nothing")

    result = await adapter.search(client, {}, req, "tavily")
    assert result.results == []
    assert result.answer is None


@pytest.mark.asyncio
async def test_tavily_omits_empty_domain_filters():
    """include_domains / exclude_domains should not appear in payload when empty."""
    adapter = TavilyAdapter()
    resp = _mock_response(TAVILY_RESPONSE)
    client = _mock_client(resp)
    req = SearchRequest(query="q")

    await adapter.search(client, {}, req, "tavily")

    payload = client.post.call_args.kwargs.get("json", {})
    assert "include_domains" not in payload
    assert "exclude_domains" not in payload


# ---------------------------------------------------------------------------
# ExaAdapter
# ---------------------------------------------------------------------------

EXA_RESPONSE = {
    "results": [
        {
            "url": "https://example.com/a",
            "title": "Example A",
            "text": "Content of A",
            "score": 0.99,
        },
        {
            "url": "https://example.com/b",
            "title": "Example B",
            "text": "Content of B",
        },
    ]
}


@pytest.mark.asyncio
async def test_exa_basic_search():
    adapter = ExaAdapter()
    client = _mock_client(_mock_response(EXA_RESPONSE))
    req = SearchRequest(query="example search")

    result = await adapter.search(client, {"x-api-key": "key"}, req, "exa")

    assert result.provider == "exa"
    assert len(result.results) == 2
    assert result.results[0].score == 0.99
    assert result.results[0].content == "Content of A"
    assert result.answer is None  # Exa has no synthesized answer


@pytest.mark.asyncio
async def test_exa_derives_score_from_position_when_missing():
    adapter = ExaAdapter()
    response_data = {
        "results": [
            {"url": "https://a.com", "title": "A", "text": "text a"},
            {"url": "https://b.com", "title": "B", "text": "text b"},
        ]
    }
    client = _mock_client(_mock_response(response_data))
    req = SearchRequest(query="q")

    result = await adapter.search(client, {}, req, "exa")

    assert result.results[0].score == pytest.approx(1.0)
    assert result.results[1].score == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_exa_passes_num_results_and_domains():
    adapter = ExaAdapter()
    client = _mock_client(_mock_response(EXA_RESPONSE))
    req = SearchRequest(
        query="q",
        num_results=7,
        include_domains=["a.com"],
        exclude_domains=["b.com"],
    )

    await adapter.search(client, {}, req, "exa")

    payload = client.post.call_args.kwargs.get("json", {})
    assert payload["numResults"] == 7
    assert payload["includeDomains"] == ["a.com"]
    assert payload["excludeDomains"] == ["b.com"]


@pytest.mark.asyncio
async def test_exa_raw_content_when_requested():
    adapter = ExaAdapter()
    response_data = {
        "results": [{"url": "https://x.com", "title": "X", "text": "full text here"}]
    }
    client = _mock_client(_mock_response(response_data))
    req = SearchRequest(query="q", include_raw_content=True)

    result = await adapter.search(client, {}, req, "exa")
    assert result.results[0].raw_content == "full text here"


# ---------------------------------------------------------------------------
# BraveAdapter
# ---------------------------------------------------------------------------

BRAVE_RESPONSE = {
    "web": {
        "results": [
            {
                "url": "https://brave.example.com/a",
                "title": "Brave Result A",
                "description": "A snippet from Brave.",
            },
            {
                "url": "https://brave.example.com/b",
                "title": "Brave Result B",
                "description": "",
                "extra_snippets": ["extra one", "extra two"],
            },
        ]
    }
}


@pytest.mark.asyncio
async def test_brave_basic_search():
    adapter = BraveAdapter()
    client = _mock_client(_mock_response(BRAVE_RESPONSE))
    req = SearchRequest(query="brave search test")

    result = await adapter.search(client, {"X-Subscription-Token": "key"}, req, "brave")

    assert result.provider == "brave"
    assert len(result.results) == 2
    assert result.results[0].content == "A snippet from Brave."
    assert result.results[0].score == pytest.approx(1.0)
    assert result.results[1].score == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_brave_falls_back_to_extra_snippets():
    adapter = BraveAdapter()
    client = _mock_client(_mock_response(BRAVE_RESPONSE))
    req = SearchRequest(query="q")

    result = await adapter.search(client, {}, req, "brave")

    # Second result has empty description, should use extra_snippets
    assert result.results[1].content == "extra one extra two"


@pytest.mark.asyncio
async def test_brave_passes_query_and_count():
    adapter = BraveAdapter()
    client = _mock_client(_mock_response(BRAVE_RESPONSE))
    req = SearchRequest(query="brave query", num_results=8)

    await adapter.search(client, {}, req, "brave")

    params = client.get.call_args.kwargs.get("params", {})
    assert params["q"] == "brave query"
    assert params["count"] == 8


@pytest.mark.asyncio
async def test_brave_empty_web_results():
    adapter = BraveAdapter()
    client = _mock_client(_mock_response({"web": {"results": []}}))
    req = SearchRequest(query="nothing")

    result = await adapter.search(client, {}, req, "brave")
    assert result.results == []
    assert result.answer is None


# ---------------------------------------------------------------------------
# get_adapter factory
# ---------------------------------------------------------------------------

def test_get_adapter_tavily():
    a = get_adapter("tavily")
    assert isinstance(a, TavilyAdapter)


def test_get_adapter_exa():
    a = get_adapter("exa")
    assert isinstance(a, ExaAdapter)


def test_get_adapter_brave():
    a = get_adapter("brave")
    assert isinstance(a, BraveAdapter)


def test_get_adapter_unknown_raises():
    with pytest.raises(ValueError, match="Unknown search provider type"):
        get_adapter("unknown-provider")
