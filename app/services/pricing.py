import asyncio
import time

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
        self._providers = providers
        self._ttl = ttl
        self._last_refresh: float = 0
        self._task: asyncio.Task | None = None

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
