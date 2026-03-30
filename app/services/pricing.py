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

    @classmethod
    def from_endpoint(
        cls,
        url: str,
        prefix: str = "proxy",
        currency: str = "USD",
        ttl: int = _DEFAULT_TTL,
    ) -> "PricingCache":
        """Create a PricingCache that fetches pricing from a remote endpoint.

        The endpoint is polled on startup and refreshed every *ttl* seconds.
        Expected response: OpenRouter-style ``{model_id: {pricing: {input, output}}}``
        or flat ``{model_id: {input_cost_per_token, output_cost_per_token}}``.
        """
        instance = cls(providers=[], ttl=ttl)
        instance._endpoint_url = url
        instance._endpoint_prefix = prefix
        instance._endpoint_currency = currency
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
        # Standalone endpoint mode (from_endpoint factory)
        endpoint_url = getattr(self, "_endpoint_url", None)
        if endpoint_url:
            await self._fetch_endpoint(
                endpoint_url,
                self._endpoint_prefix,
                self._endpoint_currency,
            )

        # Provider-based mode (LLM Hub)
        for provider in self._providers:
            config = provider.config
            prefix = config.prefix

            # Try dynamic endpoint first
            if config.pricing_endpoint:
                url = f"{config.base_url}{config.pricing_endpoint}"
                ok = await self._fetch_endpoint(
                    url, prefix, config.currency or "USD", verify_ssl=config.verify_ssl,
                )
                if ok:
                    continue

            # Fall back to static config
            if config.pricing:
                self._cache[prefix] = dict(config.pricing)
                self._currencies[prefix] = config.currency or "USD"
                logger.info(
                    f"Loaded static pricing for '{prefix}': {len(config.pricing)} models"
                )

        self._last_refresh = time.time()

    async def _fetch_endpoint(
        self,
        url: str,
        prefix: str,
        currency: str,
        verify_ssl: bool = True,
    ) -> bool:
        """Fetch pricing from a remote URL. Returns True on success."""
        try:
            async with httpx.AsyncClient(verify=verify_ssl, timeout=30) as client:
                resp = await client.get(url)
            if resp.status_code == 200:
                parsed = self._parse_pricing_response(resp.json())
                if parsed:
                    self._cache[prefix] = parsed
                    self._currencies[prefix] = currency
                    logger.info(
                        f"Loaded dynamic pricing for '{prefix}' from {url}: {len(parsed)} models"
                    )
                    return True
        except Exception as e:
            logger.warning(f"Failed to fetch pricing from {url} for '{prefix}': {e}")
        return False

    @staticmethod
    def _parse_pricing_response(data) -> dict[str, PricingInfo] | None:
        """Parse a pricing response into a PricingInfo map.

        Supports three formats:
        1. OpenRouter-style: ``{model_id: {"pricing": {"input": x, "output": y}}}``
        2. Flat dict: ``{model_id: {"input_cost_per_token": x, "output_cost_per_token": y}}``
        3. List (vsellm-style): ``[{"Public Name": "gpt-4o", "Input Cost $": 4.5, "Output Cost $": 13.5}]``
           Costs are per 1M tokens — divided by 1_000_000 to get per-token.
        """
        result: dict[str, PricingInfo] = {}

        # Format 3: list of model objects
        if isinstance(data, list):
            for item in data:
                try:
                    if not isinstance(item, dict):
                        continue
                    name = item.get("Public Name") or item.get("public_name") or item.get("name")
                    if not name:
                        continue
                    input_cost = float(item.get("Input Cost $", 0) or item.get("input_cost", 0))
                    output_cost = float(item.get("Output Cost $", 0) or item.get("output_cost", 0))
                    result[name] = PricingInfo(
                        input_cost_per_token=input_cost / 1_000_000,
                        output_cost_per_token=output_cost / 1_000_000,
                    )
                except (TypeError, ValueError, AttributeError):
                    continue
            return result or None

        # Format 1 & 2: dict keyed by model_id
        models = data if isinstance(data, dict) else {}
        for model_id, info in models.items():
            try:
                if not isinstance(info, dict):
                    continue
                # OpenRouter-style nested pricing
                if "pricing" in info:
                    pricing = info["pricing"]
                    input_cost = float(pricing.get("input", 0))
                    output_cost = float(pricing.get("output", 0))
                # Flat format
                elif "input_cost_per_token" in info:
                    input_cost = float(info["input_cost_per_token"])
                    output_cost = float(info.get("output_cost_per_token", 0))
                else:
                    continue
                result[model_id] = PricingInfo(
                    input_cost_per_token=input_cost,
                    output_cost_per_token=output_cost,
                )
            except (TypeError, ValueError, AttributeError):
                continue
        return result or None
