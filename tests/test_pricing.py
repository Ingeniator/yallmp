import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.schemas.provider import LlmProviderConfig, PricingInfo
from app.services.pricing import CostBreakdown, PricingCache
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
    assert isinstance(cost, CostBreakdown)
    assert cost.input == pytest.approx(0.001 * 100)
    assert cost.output == pytest.approx(0.002 * 50)
    assert cost.total == pytest.approx(0.001 * 100 + 0.002 * 50)


def test_get_cost_unknown_provider():
    cache = PricingCache([])
    assert cache.get_cost("unknown", "model", 10, 10) is None


def test_get_cost_unknown_model():
    provider = _make_provider(prefix="p")
    cache = PricingCache([provider])
    cache._cache["p"] = {"other-model": PricingInfo(input_cost_per_token=0.1, output_cost_per_token=0.2)}

    assert cache.get_cost("p", "missing-model", 10, 10) is None


# ---------------------------------------------------------------------------
# PricingCache.get_image_cost / find_image_cost
# ---------------------------------------------------------------------------

def test_get_image_cost_matches_size_and_quality():
    cache = PricingCache([])
    cache._cache["openai"] = {
        "dall-e-3": PricingInfo(image_cost={"1024x1024:hd": 0.08, "1024x1024:standard": 0.04}),
    }
    cache._currencies["openai"] = "USD"

    cost = cache.get_image_cost("openai", "dall-e-3", "1024x1024", "hd", count=1)
    assert cost.total == pytest.approx(0.08)
    assert cost.input == 0.0


def test_get_image_cost_multiple_images_scales_linearly():
    cache = PricingCache([])
    cache._cache["openai"] = {"dall-e-3": PricingInfo(image_cost={"1024x1024:hd": 0.08})}

    cost = cache.get_image_cost("openai", "dall-e-3", "1024x1024", "hd", count=3)
    assert cost.total == pytest.approx(0.24)


def test_get_image_cost_falls_back_to_size_only_key():
    """dall-e-2 has no quality tiers — pricing may be keyed by size alone."""
    cache = PricingCache([])
    cache._cache["openai"] = {"dall-e-2": PricingInfo(image_cost={"1024x1024": 0.02})}

    cost = cache.get_image_cost("openai", "dall-e-2", "1024x1024", "standard", count=1)
    assert cost.total == pytest.approx(0.02)


def test_get_image_cost_unknown_size_quality_returns_none():
    cache = PricingCache([])
    cache._cache["openai"] = {"dall-e-3": PricingInfo(image_cost={"1024x1024:hd": 0.08})}

    assert cache.get_image_cost("openai", "dall-e-3", "512x512", "hd", count=1) is None


def test_get_image_cost_model_without_image_cost_returns_none():
    cache = PricingCache([])
    cache._cache["openai"] = {"gpt-4o": PricingInfo(input_cost_per_token=0.001, output_cost_per_token=0.002)}

    assert cache.get_image_cost("openai", "gpt-4o", "1024x1024", "hd", count=1) is None


def test_find_image_cost_searches_all_providers():
    cache = PricingCache([])
    cache._cache["other"] = {"llama3": PricingInfo(input_cost_per_token=0.1, output_cost_per_token=0.2)}
    cache._cache["openai"] = {"dall-e-3": PricingInfo(image_cost={"1024x1024:hd": 0.08})}
    cache._currencies["openai"] = "USD"

    found = cache.find_image_cost("dall-e-3", "1024x1024", "hd", count=1)
    assert found is not None
    prefix, currency, cost = found
    assert prefix == "openai"
    assert currency == "USD"
    assert cost.total == pytest.approx(0.08)


def test_find_image_cost_no_match_returns_none():
    cache = PricingCache([])
    cache._cache["openai"] = {"dall-e-3": PricingInfo(image_cost={"1024x1024:hd": 0.08})}

    assert cache.find_image_cost("unknown-model", "1024x1024", "hd", count=1) is None


# ---------------------------------------------------------------------------
# PricingCache.get_character_cost / find_character_cost
# ---------------------------------------------------------------------------

def test_get_character_cost():
    cache = PricingCache([])
    cache._cache["openai"] = {"tts-1": PricingInfo(cost_per_character=0.000015)}
    cache._currencies["openai"] = "USD"

    cost = cache.get_character_cost("openai", "tts-1", num_characters=1000)
    assert cost.total == pytest.approx(0.015)
    assert cost.input == 0.0


def test_get_character_cost_model_without_tts_pricing_returns_none():
    cache = PricingCache([])
    cache._cache["openai"] = {"gpt-4o": PricingInfo(input_cost_per_token=0.001, output_cost_per_token=0.002)}

    assert cache.get_character_cost("openai", "gpt-4o", num_characters=1000) is None


def test_find_character_cost_searches_all_providers():
    cache = PricingCache([])
    cache._cache["other"] = {"llama3": PricingInfo(input_cost_per_token=0.1, output_cost_per_token=0.2)}
    cache._cache["openai"] = {"tts-1": PricingInfo(cost_per_character=0.000015)}
    cache._currencies["openai"] = "USD"

    found = cache.find_character_cost("tts-1", num_characters=100)
    assert found is not None
    prefix, currency, cost = found
    assert prefix == "openai"
    assert cost.total == pytest.approx(0.0015)


def test_find_character_cost_no_match_returns_none():
    cache = PricingCache([])
    cache._cache["openai"] = {"tts-1": PricingInfo(cost_per_character=0.000015)}

    assert cache.find_character_cost("unknown-model", num_characters=100) is None


# ---------------------------------------------------------------------------
# PricingCache.from_json – image_cost / cost_per_character parsing
# ---------------------------------------------------------------------------

def test_from_json_parses_image_and_character_pricing(tmp_path):
    config_path = tmp_path / "openai_pricing.json"
    config_path.write_text(
        """
        {
          "prefix": "openai",
          "currency": "USD",
          "pricing": {
            "dall-e-3": {"image_cost": {"1024x1024:hd": 0.08, "1024x1024:standard": 0.04}},
            "tts-1": {"cost_per_character": 0.000015},
            "gpt-4o": {"input_cost_per_token": 0.0025, "output_cost_per_token": 0.01}
          }
        }
        """
    )
    cache = PricingCache.from_json(str(config_path))

    dalle = cache._cache["openai"]["dall-e-3"]
    assert dalle.image_cost == {"1024x1024:hd": 0.08, "1024x1024:standard": 0.04}

    tts = cache._cache["openai"]["tts-1"]
    assert tts.cost_per_character == 0.000015

    gpt = cache._cache["openai"]["gpt-4o"]
    assert gpt.input_cost_per_token == 0.0025
    assert gpt.image_cost is None
    assert gpt.cost_per_character is None


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

@patch("app.services.metrics_callback_handler.llm_output_cost")
@patch("app.services.metrics_callback_handler.llm_input_cost")
@patch("app.services.metrics_callback_handler.llm_cost")
@patch("app.services.metrics_callback_handler.total_token_usage_counter")
@patch("app.services.metrics_callback_handler.prompt_token_usage_counter")
@patch("app.services.metrics_callback_handler.completion_token_usage_counter")
def test_handler_increments_cost_counter(comp_c, prompt_c, total_c, cost_counter, inp_cost_c, out_cost_c):
    pricing_cache = MagicMock()
    pricing_cache.get_cost.return_value = CostBreakdown(input=0.10, output=0.32, total=0.42)

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
    cost_labels = dict(provider="openrouter", currency="USD", model="gpt-4", group_id="g1")
    cost_counter.labels.assert_called_with(**cost_labels)
    cost_counter.labels.return_value.inc.assert_called_with(0.42)
    inp_cost_c.labels.assert_called_with(**cost_labels)
    inp_cost_c.labels.return_value.inc.assert_called_with(0.10)
    out_cost_c.labels.assert_called_with(**cost_labels)
    out_cost_c.labels.return_value.inc.assert_called_with(0.32)


@patch("app.services.metrics_callback_handler.llm_cost")
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


@patch("app.services.metrics_callback_handler.llm_cost")
@patch("app.services.metrics_callback_handler.total_token_usage_counter")
@patch("app.services.metrics_callback_handler.prompt_token_usage_counter")
@patch("app.services.metrics_callback_handler.completion_token_usage_counter")
def test_handler_uses_request_model_for_pricing(comp_c, prompt_c, total_c, cost_counter):
    """Cost lookup uses the request model name, not the versioned response model."""
    pricing_cache = MagicMock()
    pricing_cache.get_cost.return_value = CostBreakdown(input=0.5, output=1.0, total=1.5)

    metadata = ChainMetadataForTracking(
        chain_type=ChainType.prompt, chain_name="proxy", group_id="g1"
    )
    handler = MetricsCallbackHandler(
        metadata=metadata,
        provider_prefix="prov",
        currency="USD",
        pricing_cache=pricing_cache,
        request_model="gpt-4o",
    )

    response = {
        "usage": {"total_tokens": 200, "prompt_tokens": 120, "completion_tokens": 80},
        "model": "gpt-4o-2024-08-06",  # versioned name from response
    }
    handler.on_llm_end(response)

    # Should use request_model "gpt-4o", not response "gpt-4o-2024-08-06"
    pricing_cache.get_cost.assert_called_once_with("prov", "gpt-4o", 120, 80)


@patch("app.services.metrics_callback_handler.llm_cost")
@patch("app.services.metrics_callback_handler.total_token_usage_counter")
@patch("app.services.metrics_callback_handler.prompt_token_usage_counter")
@patch("app.services.metrics_callback_handler.completion_token_usage_counter")
def test_handler_falls_back_to_response_model_when_no_request_model(comp_c, prompt_c, total_c, cost_counter):
    """When request_model is not set, falls back to response model name."""
    pricing_cache = MagicMock()
    pricing_cache.get_cost.return_value = CostBreakdown(input=0.2, output=0.3, total=0.5)

    metadata = ChainMetadataForTracking(
        chain_type=ChainType.prompt, chain_name="proxy", group_id="g1"
    )
    handler = MetricsCallbackHandler(
        metadata=metadata,
        provider_prefix="prov",
        currency="USD",
        pricing_cache=pricing_cache,
    )

    response = {
        "usage": {"total_tokens": 100, "prompt_tokens": 60, "completion_tokens": 40},
        "model": "gpt-4o-2024-08-06",
    }
    handler.on_llm_end(response)

    # No request_model — should use response model
    pricing_cache.get_cost.assert_called_once_with("prov", "gpt-4o-2024-08-06", 60, 40)


@patch("app.services.metrics_callback_handler.llm_cost")
@patch("app.services.metrics_callback_handler.total_token_usage_counter")
@patch("app.services.metrics_callback_handler.prompt_token_usage_counter")
@patch("app.services.metrics_callback_handler.completion_token_usage_counter")
def test_handler_cost_works_with_empty_provider_prefix(comp_c, prompt_c, total_c, cost_counter):
    """Empty string provider prefix should still trigger cost calculation."""
    pricing_cache = MagicMock()
    pricing_cache.get_cost.return_value = CostBreakdown(input=0.44, output=0.55, total=0.99)

    metadata = ChainMetadataForTracking(
        chain_type=ChainType.prompt, chain_name="proxy", group_id="g1"
    )
    handler = MetricsCallbackHandler(
        metadata=metadata,
        provider_prefix="",
        currency="USD",
        pricing_cache=pricing_cache,
        request_model="gpt-5",
    )

    response = {
        "usage": {"total_tokens": 50, "prompt_tokens": 30, "completion_tokens": 20},
        "model": "gpt-5-2025-04-14",
    }
    handler.on_llm_end(response)

    pricing_cache.get_cost.assert_called_once_with("", "gpt-5", 30, 20)
    cost_counter.labels.return_value.inc.assert_called_with(0.99)


@patch("app.services.metrics_callback_handler.llm_cost")
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
