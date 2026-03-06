import asyncio
import json
import time
from pathlib import Path

import httpx

from app.core.logging_config import setup_logging
from app.schemas.provider import PricingInfo

logger = setup_logging()

_DEFAULT_TTL = 86400  # 24 hours


class PricingCache:
    """Fetches and caches per-model pricing for LLM providers."""

    def __init__(self, providers: list, ttl: int = _DEFAULT_TTL):
        # provider -> model -> PricingInfo
        self._cache: dict[str, dict[str, PricingInfo]] = {}
        # prefix -> currency (for find_cost lookups)
        self._currencies: dict[str, str] = {}
        self._providers = providers
        self._ttl = ttl
        self._last_refresh: float = 0
        self._task: asyncio.Task | None = None

    @classmethod
    def from_json(cls, path: str) -> "PricingCache":
        """Create a PricingCache from a static JSON config file.

        Expected format (same as llm_hub provider config):
        {
          "prefix": "myproxy",
          "currency": "USD",
          "pricing": {
            "model-name": {"input_cost_per_token": 0.001, "output_cost_per_token": 0.002}
          }
        }
        """
        instance = cls(providers=[])
        data = json.loads(Path(path).read_text())
        prefix = data.get("prefix", "proxy")
        currency = data.get("currency", "USD")
        pricing_raw = data.get("pricing", {})

        models: dict[str, PricingInfo] = {}
        for model_id, costs in pricing_raw.items():
            models[model_id] = PricingInfo(
                input_cost_per_token=float(costs.get("input_cost_per_token", 0)),
                output_cost_per_token=float(costs.get("output_cost_per_token", 0)),
            )

        if models:
            instance._cache[prefix] = models
            instance._currencies[prefix] = currency
            logger.info(f"Loaded static pricing for '{prefix}': {len(models)} models")

        return instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def startup(self):
        await self._refresh()
        self._task = asyncio.create_task(self._background_loop())

    async def shutdown(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def get_cost(
        self,
        provider_prefix: str,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float | None:
        models = self._cache.get(provider_prefix)
        if not models:
            return None
        pricing = models.get(model_name)
        if not pricing:
            return None
        return (
            prompt_tokens * pricing.input_cost_per_token
            + completion_tokens * pricing.output_cost_per_token
        )

    def find_cost(
        self,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> tuple[str, str, float] | None:
        """Search all providers for pricing and return (prefix, currency, cost) or None."""
        for prefix in self._cache:
            cost = self.get_cost(prefix, model_name, prompt_tokens, completion_tokens)
            if cost is not None:
                currency = self._currencies.get(prefix, "USD")
                return prefix, currency, cost
        return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _background_loop(self):
        while True:
            await asyncio.sleep(self._ttl)
            await self._refresh()

    async def _refresh(self):
        for provider in self._providers:
            config = provider.config
            prefix = config.prefix

            # Try dynamic endpoint first
            if config.pricing_endpoint:
                try:
                    url = f"{config.base_url}{config.pricing_endpoint}"
                    async with httpx.AsyncClient(verify=config.verify_ssl, timeout=30) as client:
                        resp = await client.get(url)
                    if resp.status_code == 200:
                        parsed = self._parse_pricing_response(resp.json())
                        if parsed:
                            self._cache[prefix] = parsed
                            self._currencies[prefix] = config.currency or "USD"
                            logger.info(
                                f"Loaded dynamic pricing for '{prefix}': {len(parsed)} models"
                            )
                            continue
                except Exception as e:
                    logger.warning(
                        f"Failed to fetch pricing from {config.pricing_endpoint} "
                        f"for '{prefix}': {e}"
                    )

            # Fall back to static config
            if config.pricing:
                self._cache[prefix] = dict(config.pricing)
                self._currencies[prefix] = config.currency or "USD"
                logger.info(
                    f"Loaded static pricing for '{prefix}': {len(config.pricing)} models"
                )

        self._last_refresh = time.time()

    @staticmethod
    def _parse_pricing_response(data: dict) -> dict[str, PricingInfo] | None:
        """Parse an OpenRouter-style /v1/models_info response into PricingInfo map."""
        result: dict[str, PricingInfo] = {}
        models = data if isinstance(data, dict) else {}
        for model_id, info in models.items():
            try:
                pricing = info.get("pricing", {})
                input_cost = float(pricing.get("input", 0))
                output_cost = float(pricing.get("output", 0))
                result[model_id] = PricingInfo(
                    input_cost_per_token=input_cost,
                    output_cost_per_token=output_cost,
                )
            except (TypeError, ValueError, AttributeError):
                continue
        return result or None
