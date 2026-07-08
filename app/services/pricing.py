import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from app.core.logging_config import setup_logging
from app.schemas.provider import PricingInfo

logger = setup_logging()


@dataclass
class CostBreakdown:
    """Per-call cost split into input (prompt) and output (completion) components."""

    input: float
    output: float
    total: float

    def __bool__(self) -> bool:
        return self.total != 0


_DEFAULT_TTL = 86400  # 24 hours


class PricingCache:
    """Fetches and caches per-model pricing for LLM providers."""

    def __init__(
        self,
        providers: dict,
        ttl: int = _DEFAULT_TTL,
        resolve_prefix: Callable[[str], str | None] | None = None,
    ):
        # model name -> prefix of the provider the proxy routes it to
        # (LlmHub routing). Lets find_cost price a model at the provider that
        # actually served it, e.g. the default provider for unprefixed models.
        self._resolve_prefix = resolve_prefix
        # provider -> model -> PricingInfo
        self._cache: dict[str, dict[str, PricingInfo]] = {}
        # prefix -> currency (for find_cost lookups)
        self._currencies: dict[str, str] = {}
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

    def _lookup(self, provider_prefix: str, model_name: str) -> PricingInfo | None:
        """Pricing for *model_name* at *provider_prefix*, or None. Does not log.

        Tries the name as given, then without this provider's own
        ``<prefix>/``. Model ids may contain slashes themselves
        (``openrouter/anthropic/claude-sonnet`` -> ``anthropic/claude-sonnet``),
        so only the provider's prefix is removed, and only when it is present.
        """
        models = self._cache.get(provider_prefix)
        if not models:
            return None
        pricing = models.get(model_name)
        own = f"{provider_prefix}/"
        if pricing is None and model_name.startswith(own):
            pricing = models.get(model_name[len(own):])
        return pricing

    @staticmethod
    def _breakdown(
        pricing: PricingInfo, prompt_tokens: int, completion_tokens: int
    ) -> CostBreakdown:
        input_cost = prompt_tokens * pricing.input_cost_per_token
        output_cost = completion_tokens * pricing.output_cost_per_token
        return CostBreakdown(
            input=input_cost, output=output_cost, total=input_cost + output_cost
        )

    def get_cost(
        self,
        provider_prefix: str,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> CostBreakdown | None:
        if not self._cache.get(provider_prefix):
            logger.warning("get_cost: no cache for provider", provider=provider_prefix)
            return None
        pricing = self._lookup(provider_prefix, model_name)
        if not pricing:
            logger.error(
                "get_cost: model not found in pricing",
                model=model_name,
                provider=provider_prefix,
            )
            return None
        return self._breakdown(pricing, prompt_tokens, completion_tokens)

    def find_cost(
        self,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> tuple[str, str, CostBreakdown] | None:
        """Price a model whose provider is not known; return (prefix, currency, cost) or None.

        With a ``resolve_prefix`` routing function, the model is priced only at
        the provider it routes to (``prefix/model`` -> that provider, a
        ``models`` list match, else the default provider), never at another
        provider's rate.

        Without one: a ``<prefix>/model`` name naming a cached provider is priced
        by that provider only. Otherwise the model must be priced by exactly one
        provider, or identically (same prices and currency) by all providers
        that list it. Conflicting prices are ambiguous: we return None rather
        than charge another provider's rate.
        """
        if self._resolve_prefix is not None:
            prefix = self._resolve_prefix(model_name)
            pricing = self._lookup(prefix, model_name) if prefix else None
            if pricing is None:
                logger.warning(
                    "find_cost: no pricing at routed provider",
                    model=model_name,
                    provider=prefix,
                )
                return None
            return (
                prefix,
                self._currencies.get(prefix, "USD"),
                self._breakdown(pricing, prompt_tokens, completion_tokens),
            )

        head = model_name.split("/", 1)[0]
        if "/" in model_name and head in self._cache:
            candidates = [head]
        else:
            candidates = list(self._cache)

        matches = [
            (prefix, self._currencies.get(prefix, "USD"), pricing)
            for prefix in candidates
            if (pricing := self._lookup(prefix, model_name)) is not None
        ]
        if not matches:
            return None

        distinct = {
            (cur, p.input_cost_per_token, p.output_cost_per_token)
            for _, cur, p in matches
        }
        if len(distinct) > 1:
            logger.warning(
                "find_cost: ambiguous pricing, not charging",
                model=model_name,
                providers=[prefix for prefix, _, _ in matches],
            )
            return None

        prefix, currency, pricing = matches[0]
        return prefix, currency, self._breakdown(
            pricing, prompt_tokens, completion_tokens
        )

    def _find_unit_cost(
        self,
        model_name: str,
        price: Callable[[str], CostBreakdown | None],
    ) -> tuple[str, str, CostBreakdown] | None:
        """Provider routing for non-token costs, same rules as ``find_cost``:
        the routed provider when ``resolve_prefix`` is set, else the
        ``<prefix>/`` provider, else a unique (or identically priced) match."""
        if self._resolve_prefix is not None:
            prefix = self._resolve_prefix(model_name)
            cost = price(prefix) if prefix else None
            if cost is None:
                return None
            return prefix, self._currencies.get(prefix, "USD"), cost

        head = model_name.split("/", 1)[0]
        if "/" in model_name and head in self._cache:
            candidates = [head]
        else:
            candidates = list(self._cache)

        matches = [
            (prefix, self._currencies.get(prefix, "USD"), cost)
            for prefix in candidates
            if (cost := price(prefix)) is not None
        ]
        if not matches:
            return None
        if len({(cur, c.total) for _, cur, c in matches}) > 1:
            logger.warning(
                "find_unit_cost: ambiguous pricing, not charging",
                model=model_name,
                providers=[prefix for prefix, _, _ in matches],
            )
            return None
        return matches[0]

    def get_image_cost(
        self,
        provider_prefix: str,
        model_name: str,
        size: str,
        quality: str,
        count: int = 1,
    ) -> CostBreakdown | None:
        """Per-image cost for image-generation models (e.g. dall-e-3). The whole
        amount is booked as `output` — there's no prompt/completion split for a
        generated image the way there is for tokens."""
        pricing = self._lookup(provider_prefix, model_name)
        if not pricing or not pricing.image_cost:
            return None
        per_image = pricing.image_cost.get(f"{size}:{quality}")
        if per_image is None:
            per_image = pricing.image_cost.get(size)
        if per_image is None:
            return None
        total = per_image * count
        return CostBreakdown(input=0.0, output=total, total=total)

    def find_image_cost(
        self,
        model_name: str,
        size: str,
        quality: str,
        count: int = 1,
    ) -> tuple[str, str, CostBreakdown] | None:
        return self._find_unit_cost(
            model_name,
            lambda prefix: self.get_image_cost(prefix, model_name, size, quality, count),
        )

    def get_character_cost(
        self,
        provider_prefix: str,
        model_name: str,
        num_characters: int,
    ) -> CostBreakdown | None:
        """Per-character cost for TTS models. Booked entirely as `output`."""
        pricing = self._lookup(provider_prefix, model_name)
        if not pricing or pricing.cost_per_character is None:
            return None
        total = pricing.cost_per_character * num_characters
        return CostBreakdown(input=0.0, output=total, total=total)

    def find_character_cost(
        self,
        model_name: str,
        num_characters: int,
    ) -> tuple[str, str, CostBreakdown] | None:
        return self._find_unit_cost(
            model_name,
            lambda prefix: self.get_character_cost(prefix, model_name, num_characters),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _background_loop(self):
        while True:
            await asyncio.sleep(self._ttl)
            await self._refresh()

    async def _refresh(self):
        for _, provider in self._providers.items():
            config = provider.config
            prefix = config.prefix

            # Try dynamic endpoint first
            if config.pricing_endpoint:
                url = f"{config.base_url}{config.pricing_endpoint}"
                ok = await self._fetch_endpoint(
                    url,
                    prefix,
                    config.currency or "USD",
                    verify_ssl=config.verify_ssl,
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
                else:
                    logger.error("_fetch_endpoint: Parsed price with error")
            else:
                error = resp.json()
                logger.error(
                    f"Get price from {url}: status {resp.status_code}. Error - {error}"
                )
        except Exception as e:
            logger.error(f"Failed to fetch pricing from {url} for '{prefix}': {e}")
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
                    name = (
                        item.get("Public Name")
                        or item.get("public_name")
                        or item.get("name")
                        or item.get("Model")
                    )
                    if not name:
                        continue
                    input_cost = float(
                        item.get("Input Cost $", 0) or item.get("input_cost", 0)
                    )
                    output_cost = float(
                        item.get("Output Cost $", 0) or item.get("output_cost", 0)
                    )
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
