"""Tests for search provider adapters.

All HTTP calls are mocked — no real network traffic.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.schemas.search import SearchRequest, SearchResponse
from app.services.search_adapters.tavily import TavilyAdapter
from app.services.search_adapters.exa import ExaAdapter
from app.services.search_adapters.brave import BraveAdapter
from app.services.search_adapters.yandex import YandexAdapter
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

    result = await adapter.search(
        client, {"Authorization": "Bearer key"}, req, "tavily"
    )

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
# YandexAdapter
# ---------------------------------------------------------------------------

YANDEX_XML = """<?xml version="1.0" encoding="utf-8"?>
<yandexsearch version="1.0">
<response date="20260807T154024">
<results>
<grouping attr="d">
<group>
<doc id="Z1">
<url>https://translate.yandex.ru/hi</url>
<domain>translate.yandex.ru</domain>
<title><hlword>Hi</hlword> - \u043e\u043d\u043b\u0430\u0439\u043d \u043f\u0435\u0440\u0435\u0432\u043e\u0434</title>
<passages>
<passage>\u041f\u0435\u0440\u0435\u0432\u043e\u0434 \u0441\u043b\u043e\u0432\u0430 "<hlword>Hi</hlword>" \u0441 \u0430\u043d\u0433\u043b\u0438\u0439\u0441\u043a\u043e\u0433\u043e.</passage>
<passage>\u0412\u0442\u043e\u0440\u043e\u0439 \u043f\u0430\u0441\u0441\u0430\u0436.</passage>
</passages>
<properties>
<extended-text>hi. \u0411\u044b\u0441\u0442\u0440\u044b\u0439 \u043f\u0435\u0440\u0435\u0432\u043e\u0434.</extended-text>
</properties>
</doc>
</group>
<group>
<doc id="Z2">
<url>https://www.wordreference.com/enru/hi</url>
<domain>www.wordreference.com</domain>
<title><hlword>hi</hlword> - \u0410\u043d\u0433\u043b\u043e-\u0440\u0443\u0441\u0441\u043a\u0438\u0439 \u0441\u043b\u043e\u0432\u0430\u0440\u044c</title>
<headline><hlword>hi</hlword> - \u0411\u0435\u0441\u043f\u043b\u0430\u0442\u043d\u044b\u0439 \u043e\u043d\u043b\u0430\u0439\u043d \u0441\u043b\u043e\u0432\u0430\u0440\u044c.</headline>
<properties>
<extended-text>hi. [links].</extended-text>
</properties>
</doc>
</group>
</grouping>
</results>
</response>
</yandexsearch>
"""

YANDEX_RESPONSE = {"raw_text": YANDEX_XML}


@pytest.mark.asyncio
async def test_yandex_basic_search():
    adapter = YandexAdapter()
    client = _mock_client(_mock_response(YANDEX_RESPONSE))
    req = SearchRequest(query="hi")

    result = await adapter.search(client, {"X-API-Key": "key"}, req, "yandex")

    assert result.provider == "yandex"
    assert result.query == "hi"
    assert result.answer is None
    assert len(result.results) == 2
    assert result.results[0].url == "https://translate.yandex.ru/hi"
    # <hlword> markup is stripped, inner text preserved
    assert result.results[0].title == "Hi - онлайн перевод"
    assert result.results[0].content == 'Перевод слова "Hi" с английского. Второй пассаж.'
    assert result.results[0].score == pytest.approx(1.0)
    assert result.results[1].score == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_yandex_falls_back_to_headline():
    adapter = YandexAdapter()
    client = _mock_client(_mock_response(YANDEX_RESPONSE))
    req = SearchRequest(query="hi")

    result = await adapter.search(client, {}, req, "yandex")

    # Second doc has no <passages>, should use <headline>
    assert result.results[1].content == "hi - Бесплатный онлайн словарь."


@pytest.mark.asyncio
async def test_yandex_raw_content_only_when_requested():
    adapter = YandexAdapter()
    req = SearchRequest(query="hi")

    result = await adapter.search(
        _mock_client(_mock_response(YANDEX_RESPONSE)), {}, req, "yandex"
    )
    assert result.results[0].raw_content is None

    req_raw = SearchRequest(query="hi", include_raw_content=True)
    result_raw = await adapter.search(
        _mock_client(_mock_response(YANDEX_RESPONSE)), {}, req_raw, "yandex"
    )
    assert result_raw.results[0].raw_content == "hi. Быстрый перевод."


@pytest.mark.asyncio
async def test_yandex_builds_payload():
    adapter = YandexAdapter()
    client = _mock_client(_mock_response(YANDEX_RESPONSE))
    req = SearchRequest(query="python asyncio", num_results=7)

    await adapter.search(client, {}, req, "yandex")

    payload = client.post.call_args.kwargs.get("json", {})
    assert payload["query"] == {
        "searchType": "SEARCH_TYPE_RU",
        "queryText": "python asyncio",
    }
    assert payload["groupSpec"]["groupsOnPage"] == "7"


@pytest.mark.asyncio
async def test_yandex_ignores_domain_filters():
    adapter = YandexAdapter()
    client = _mock_client(_mock_response(YANDEX_RESPONSE))
    req = SearchRequest(
        query="q",
        include_domains=["a.com"],
        exclude_domains=["b.com"],
    )

    await adapter.search(client, {}, req, "yandex")

    payload = client.post.call_args.kwargs.get("json", {})
    # Yandex has no domain filter parameters; the query must stay untouched.
    assert payload["query"]["queryText"] == "q"
    assert "includeDomains" not in payload
    assert "excludeDomains" not in payload


@pytest.mark.asyncio
async def test_yandex_empty_raw_text():
    adapter = YandexAdapter()
    client = _mock_client(_mock_response({"raw_text": ""}))
    req = SearchRequest(query="nothing")

    result = await adapter.search(client, {}, req, "yandex")

    assert result.results == []
    assert result.answer is None


@pytest.mark.asyncio
async def test_yandex_rejects_doctype():
    adapter = YandexAdapter()
    bomb = (
        '<?xml version="1.0"?><!DOCTYPE l [<!ENTITY a "aaaaaaaaaa">]>'
        "<yandexsearch>&a;</yandexsearch>"
    )
    client = _mock_client(_mock_response({"raw_text": bomb}))
    req = SearchRequest(query="q")

    with pytest.raises(ValueError, match="DOCTYPE"):
        await adapter.search(client, {}, req, "yandex")


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


def test_get_adapter_yandex():
    a = get_adapter("yandex")
    assert isinstance(a, YandexAdapter)


def test_get_adapter_unknown_raises():
    with pytest.raises(ValueError, match="Unknown search provider type"):
        get_adapter("unknown-provider")
