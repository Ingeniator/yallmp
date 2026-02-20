import pytest
from unittest.mock import patch, MagicMock
from app.services.metrics_callback_handler import MetricsCallbackHandler
from app.schemas.prompt import ChainMetadataForTracking, ChainType


@pytest.fixture
def handler():
    metadata = ChainMetadataForTracking(
        chain_type=ChainType.chain, chain_name="test-chain", group_id="grp-1"
    )
    return MetricsCallbackHandler(metadata=metadata)


@patch("app.services.metrics_callback_handler.total_token_usage_counter")
@patch("app.services.metrics_callback_handler.prompt_token_usage_counter")
@patch("app.services.metrics_callback_handler.completion_token_usage_counter")
def test_on_llm_end_langchain_style(comp_counter, prompt_counter, total_counter, handler):
    token_usage = MagicMock()
    token_usage.total_tokens = 100
    token_usage.prompt_tokens = 60
    token_usage.completion_tokens = 40

    response = MagicMock()
    response.llm_output = {"token_usage": token_usage, "model_name": "gpt-test"}
    # Make dict-style access fail so only the hasattr branch fires
    response.__contains__ = lambda self, key: False

    handler.on_llm_end(response)

    total_counter.labels.return_value.inc.assert_called_with(100)
    prompt_counter.labels.return_value.inc.assert_called_with(60)
    comp_counter.labels.return_value.inc.assert_called_with(40)


@patch("app.services.metrics_callback_handler.total_token_usage_counter")
@patch("app.services.metrics_callback_handler.prompt_token_usage_counter")
@patch("app.services.metrics_callback_handler.completion_token_usage_counter")
def test_on_llm_end_proxy_dict(comp_counter, prompt_counter, total_counter, handler):
    response = {
        "usage": {"total_tokens": 200, "prompt_tokens": 120, "completion_tokens": 80},
        "model": "proxy-model",
    }

    handler.on_llm_end(response)

    total_counter.labels.return_value.inc.assert_called_with(200)
    prompt_counter.labels.return_value.inc.assert_called_with(120)
    comp_counter.labels.return_value.inc.assert_called_with(80)
    total_counter.labels.assert_called_with(
        type="chain", name="test-chain", group_id="grp-1", model="proxy-model"
    )


@patch("app.services.metrics_callback_handler.total_token_usage_counter")
@patch("app.services.metrics_callback_handler.prompt_token_usage_counter")
@patch("app.services.metrics_callback_handler.completion_token_usage_counter")
def test_on_llm_end_no_metadata(comp_counter, prompt_counter, total_counter):
    handler = MetricsCallbackHandler(metadata=None)
    response = {"usage": {"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5}, "model": "m"}
    handler.on_llm_end(response)

    total_counter.labels.assert_called_with(
        type="unknown", name="unknown", group_id="unknown", model="m"
    )
