import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.schemas.provider import LlmProviderConfig, PricingInfo
from app.services.pricing import PricingCache
from app.services.metrics_callback_handler import MetricsCallbackHandler
from app.schemas.prompt import ChainMetadataForTracking, ChainType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_provider(prefix="prov", pricing_endpoint=None, pricing=None, currency="USD"):
    config = LlmProviderConfig(
        prefix=prefix,
        base_url="http://fake",
        currency=currency,
        pricing_endpoint=pricing_endpoint,
        pricing=pricing,
    )
    provider = MagicMock()
    provider.config = config
    return provider


# ---------------------------------------------------------------------------
# PricingCache._parse_pricing_response
# ---------------------------------------------------------------------------

def test_parse_pricing_response_valid():
    data = {
        "model-a": {"pricing": {"input": "0.001", "output": "0.002"}},
        "model-b": {"pricing": {"input": 0.0005, "output": 0.001}},
    }
    result = PricingCache._parse_pricing_response(data)
    assert result is not None
    assert "model-a" in result
    assert result["model-a"].input_cost_per_token == 0.001
    assert result["model-a"].output_cost_per_token == 0.002
    assert result["model-b"].input_cost_per_token == 0.0005


def test_parse_pricing_response_empty():
    assert PricingCache._parse_pricing_response({}) is None


def test_parse_pricing_response_malformed_entries():
    data = {
        "good": {"pricing": {"input": 0.01, "output": 0.02}},
        "bad": {"pricing": {"input": "not-a-number", "output": 0.0}},
        "missing": {},
    }
    result = PricingCache._parse_pricing_response(data)
    assert result is not None
    assert "good" in result
    # "bad" should be skipped due to ValueError, "missing" has no pricing key -> defaults to 0
    # Actually float("not-a-number") raises ValueError, so it's skipped


# ---------------------------------------------------------------------------
# PricingCache.get_cost
# ---------------------------------------------------------------------------

def test_get_cost_returns_correct_value():
    provider = _make_provider(prefix="test")
    cache = PricingCache([provider])
    cache._cache["test"] = {
        "my-model": PricingInfo(input_cost_per_token=0.001, output_cost_per_token=0.002),
    }

    cost = cache.get_cost("test", "my-model", prompt_tokens=100, completion_tokens=50)
    assert cost == pytest.approx(0.001 * 100 + 0.002 * 50)


def test_get_cost_unknown_provider():
    cache = PricingCache([])
    assert cache.get_cost("unknown", "model", 10, 10) is None


def test_get_cost_unknown_model():
    provider = _make_provider(prefix="p")
    cache = PricingCache([provider])
    cache._cache["p"] = {"other-model": PricingInfo(input_cost_per_token=0.1, output_cost_per_token=0.2)}

    assert cache.get_cost("p", "missing-model", 10, 10) is None


# ---------------------------------------------------------------------------
# PricingCache._refresh – static fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_refresh_loads_static_pricing():
    static = {"m1": PricingInfo(input_cost_per_token=0.01, output_cost_per_token=0.02)}
    provider = _make_provider(prefix="static_prov", pricing=static, currency="RUB")

    cache = PricingCache([provider])
    await cache._refresh()

    assert "static_prov" in cache._cache
    assert cache._cache["static_prov"]["m1"].input_cost_per_token == 0.01


# ---------------------------------------------------------------------------
# PricingCache._refresh – dynamic endpoint
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_refresh_fetches_dynamic_endpoint():
    provider = _make_provider(prefix="dyn", pricing_endpoint="/v1/models_info")

    api_response = {
        "model-x": {"pricing": {"input": 0.005, "output": 0.01}},
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = api_response

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_resp
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("app.services.pricing.httpx.AsyncClient", return_value=mock_client):
        cache = PricingCache([provider])
        await cache._refresh()

    assert "dyn" in cache._cache
    assert "model-x" in cache._cache["dyn"]
    assert cache._cache["dyn"]["model-x"].input_cost_per_token == 0.005


@pytest.mark.asyncio
async def test_refresh_falls_back_on_endpoint_failure():
    static = {"fallback-m": PricingInfo(input_cost_per_token=0.1, output_cost_per_token=0.2)}
    provider = _make_provider(prefix="fb", pricing_endpoint="/v1/pricing", pricing=static)

    mock_client = AsyncMock()
    mock_client.get.side_effect = Exception("connection refused")
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("app.services.pricing.httpx.AsyncClient", return_value=mock_client):
        cache = PricingCache([provider])
        await cache._refresh()

    assert "fb" in cache._cache
    assert "fallback-m" in cache._cache["fb"]


# ---------------------------------------------------------------------------
# PricingCache TTL
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_refresh_updates_last_refresh_time():
    provider = _make_provider()
    cache = PricingCache([provider], ttl=60)

    assert cache._last_refresh == 0
    await cache._refresh()
    assert cache._last_refresh > 0


# ---------------------------------------------------------------------------
# MetricsCallbackHandler – cost counter
# ---------------------------------------------------------------------------

@patch("app.services.metrics_callback_handler.llm_cost_total")
@patch("app.services.metrics_callback_handler.total_token_usage_counter")
@patch("app.services.metrics_callback_handler.prompt_token_usage_counter")
@patch("app.services.metrics_callback_handler.completion_token_usage_counter")
def test_handler_increments_cost_counter(comp_c, prompt_c, total_c, cost_counter):
    pricing_cache = MagicMock()
    pricing_cache.get_cost.return_value = 0.42

    metadata = ChainMetadataForTracking(
        chain_type=ChainType.prompt, chain_name="proxy", group_id="g1"
    )
    handler = MetricsCallbackHandler(
        metadata=metadata,
        provider_prefix="openrouter",
        currency="USD",
        pricing_cache=pricing_cache,
    )

    response = {
        "usage": {"total_tokens": 150, "prompt_tokens": 100, "completion_tokens": 50},
        "model": "gpt-4",
    }
    handler.on_llm_end(response)

    pricing_cache.get_cost.assert_called_once_with("openrouter", "gpt-4", 100, 50)
    cost_counter.labels.assert_called_with(
        provider="openrouter", currency="USD", model="gpt-4", group_id="g1"
    )
    cost_counter.labels.return_value.inc.assert_called_with(0.42)


@patch("app.services.metrics_callback_handler.llm_cost_total")
@patch("app.services.metrics_callback_handler.total_token_usage_counter")
@patch("app.services.metrics_callback_handler.prompt_token_usage_counter")
@patch("app.services.metrics_callback_handler.completion_token_usage_counter")
def test_handler_skips_cost_when_no_pricing(comp_c, prompt_c, total_c, cost_counter):
    metadata = ChainMetadataForTracking(chain_type=ChainType.prompt)
    handler = MetricsCallbackHandler(metadata=metadata)

    response = {
        "usage": {"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5},
        "model": "m",
    }
    handler.on_llm_end(response)

    cost_counter.labels.assert_not_called()


@patch("app.services.metrics_callback_handler.llm_cost_total")
@patch("app.services.metrics_callback_handler.total_token_usage_counter")
@patch("app.services.metrics_callback_handler.prompt_token_usage_counter")
@patch("app.services.metrics_callback_handler.completion_token_usage_counter")
def test_handler_skips_cost_when_get_cost_returns_none(comp_c, prompt_c, total_c, cost_counter):
    pricing_cache = MagicMock()
    pricing_cache.get_cost.return_value = None

    metadata = ChainMetadataForTracking(chain_type=ChainType.prompt, group_id="g")
    handler = MetricsCallbackHandler(
        metadata=metadata,
        provider_prefix="p",
        currency="USD",
        pricing_cache=pricing_cache,
    )
    response = {
        "usage": {"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5},
        "model": "m",
    }
    handler.on_llm_end(response)

    cost_counter.labels.assert_not_called()
