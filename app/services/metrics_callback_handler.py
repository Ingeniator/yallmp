
from app.core.logging_config import setup_logging
from prometheus_client import Counter
from langchain_core.callbacks.base import BaseCallbackHandler
from app.schemas.prompt import ChainMetadataForTracking

logger = setup_logging()

total_token_usage_counter = Counter(
    'llm_total_token_usage',
    'Total LLM tokens used by the prompt',
    ["type", "name", "group_id", "model"]
)
prompt_token_usage_counter = Counter(
    'llm_prompt_token_usage',
    'Prompt LLM tokens used by the prompt',
    ["type", "name", "group_id", "model"]
)
completion_token_usage_counter = Counter(
    'llm_completion_token_usage',
    'Completion LLM tokens used by the prompt',
    ["type", "name", "group_id", "model"]
)
llm_cost_total = Counter(
    'llm_cost_total',
    'Total estimated cost in provider currency',
    ["provider", "currency", "model", "group_id"]
)


class MetricsCallbackHandler(BaseCallbackHandler):
    def __init__(
        self,
        metadata: ChainMetadataForTracking | None = None,
        provider_prefix: str | None = None,
        currency: str | None = None,
        pricing_cache=None,
    ):
        self.metadata = metadata
        self.provider_prefix = provider_prefix
        self.currency = currency
        self.pricing_cache = pricing_cache

    def on_llm_end(self, response, **kwargs):
        group_id = self.metadata.group_id if self.metadata else "unknown"
        chain_name = self.metadata.chain_name if self.metadata else "unknown"
        chain_type = self.metadata.chain_type.value if self.metadata and self.metadata.chain_type else "unknown"
        total_token_usage = 0
        prompt_token_usage = 0
        completion_token_usage = 0
        model_name = "unknown"
        logger.debug(response)

        if hasattr(response, 'llm_output') \
            and isinstance(response.llm_output, dict) \
            and 'token_usage' in response.llm_output:
                token_usage = response.llm_output.get("token_usage")
                model_name = response.llm_output.get("model_name", "unknown")
                if token_usage:
                    total_token_usage = getattr(token_usage, 'total_tokens', 0) or 0
                    prompt_token_usage = getattr(token_usage, 'prompt_tokens', 0) or 0
                    completion_token_usage = getattr(token_usage, 'completion_tokens', 0) or 0
        elif isinstance(response, dict) and "usage" in response and "model" in response:
            token_usage = response.get("usage", {})
            model_name = response.get("model", "unknown")
            total_token_usage = token_usage.get("total_tokens", 0)
            prompt_token_usage = token_usage.get("prompt_tokens", 0)
            completion_token_usage = token_usage.get("completion_tokens", 0)

        labels = dict(type=chain_type, name=chain_name, group_id=group_id, model=model_name)
        total_token_usage_counter.labels(**labels).inc(total_token_usage)
        prompt_token_usage_counter.labels(**labels).inc(prompt_token_usage)
        completion_token_usage_counter.labels(**labels).inc(completion_token_usage)

        # Cost tracking
        if self.provider_prefix and self.pricing_cache and self.currency:
            cost = self.pricing_cache.get_cost(
                self.provider_prefix, model_name, prompt_token_usage, completion_token_usage,
            )
            if cost is not None:
                llm_cost_total.labels(
                    provider=self.provider_prefix,
                    currency=self.currency,
                    model=model_name,
                    group_id=group_id,
                ).inc(cost)
