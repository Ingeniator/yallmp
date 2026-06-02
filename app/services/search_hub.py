from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from httpx import AsyncClient, Limits, Timeout
from fastapi import HTTPException

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.core.proxy import CircuitBreaker
from app.schemas.provider import AuthType
from app.schemas.search import SearchProviderConfig, SearchRequest, SearchResponse
from app.services.search_adapters import get_adapter, SearchAdapter

logger = setup_logging()


class SearchProvider:
    """Runtime state for a single search provider."""

    def __init__(self, config: SearchProviderConfig):
        self.config = config
        self.client: AsyncClient | None = None
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.failure_threshold,
            recovery_time=config.recovery_time,
            window_size=config.window_size,
        )
        self.adapter: SearchAdapter = get_adapter(config.type)

    async def startup(self) -> None:
        t = self.config.timeout
        self.client = AsyncClient(
            base_url=self.config.base_url,
            timeout=Timeout(
                connect=t.connect,
                read=t.read,
                write=t.write,
                pool=t.pool,
            ),
            limits=Limits(
                max_connections=settings.max_connections,
                max_keepalive_connections=settings.max_keepalive_connections,
            ),
        )

    async def shutdown(self) -> None:
        if self.client:
            await self.client.aclose()

    async def get_auth_headers(self) -> dict[str, str]:
        auth = self.config.auth
        if auth.type == AuthType.APIKEY and auth.api_key:
            provider_type = self.config.type
            if provider_type == "tavily":
                return {"X-Tvly-Api-Key": auth.api_key}
            if provider_type == "exa":
                return {"x-api-key": auth.api_key}
            if provider_type == "brave":
                return {"X-Subscription-Token": auth.api_key}
            # Generic fallback
            return {"X-API-KEY": auth.api_key}
        return {}

    async def search(self, req: SearchRequest) -> SearchResponse:
        """Execute a search with circuit-breaker protection."""
        if await self.circuit_breaker.check_open():
            raise HTTPException(
                status_code=503,
                detail=f"Search provider '{self.config.name}' circuit breaker open. Try later.",
            )

        assert self.client is not None, "SearchProvider not started"

        try:
            auth_headers = await self.get_auth_headers()
            response = await self.adapter.search(
                client=self.client,
                auth_headers=auth_headers,
                req=req,
                provider_name=self.config.name,
            )
            await self.circuit_breaker.record_success()
            return response
        except HTTPException:
            raise
        except Exception as exc:
            activated = await self.circuit_breaker.record_failure()
            if activated:
                logger.error(
                    "Search circuit breaker activated",
                    provider=self.config.name,
                    exc_info=exc,
                )
                raise HTTPException(
                    status_code=503,
                    detail=f"Search provider '{self.config.name}' circuit breaker activated.",
                ) from exc
            raise HTTPException(
                status_code=502,
                detail=f"Search provider '{self.config.name}' request failed: {exc}",
            ) from exc


class SearchHub:
    """Registry of search providers loaded from JSON config files."""

    def __init__(self) -> None:
        self.providers: dict[str, SearchProvider] = {}
        self._default: str | None = None

    def load_providers(self, directory: str | None = None) -> None:
        directory = directory or settings.search_hub_directory
        hub_path = Path(directory)
        if not hub_path.is_dir():
            logger.warning("Search hub directory not found", path=str(hub_path))
            return

        seen: set[str] = set()
        for json_file in sorted(hub_path.glob("*.json")):
            try:
                raw = json_file.read_text(encoding="utf-8")
                expanded = os.path.expandvars(raw)
                data = json.loads(expanded)

                if "name" not in data:
                    logger.debug("Skipping file (no 'name' field)", file=json_file.name)
                    continue

                config = SearchProviderConfig(**data)

                if config.name in seen:
                    logger.error(
                        "Duplicate search provider name — skipping",
                        name=config.name,
                        file=json_file.name,
                    )
                    continue

                seen.add(config.name)
                self.providers[config.name] = SearchProvider(config)

                if config.default:
                    if self._default is not None:
                        logger.warning(
                            "Multiple default search providers — keeping first",
                            first=self._default,
                            ignored=config.name,
                        )
                    else:
                        self._default = config.name

                logger.info(
                    "Loaded search provider",
                    name=config.name,
                    type=config.type,
                    default=config.default,
                )

            except Exception as exc:
                logger.error(
                    "Failed to load search provider",
                    file=json_file.name,
                    exc_info=exc,
                )

    async def startup(self) -> None:
        for name, provider in self.providers.items():
            try:
                await provider.startup()
                logger.info("Search provider started", name=name)
            except Exception as exc:
                logger.error("Failed to start search provider", name=name, exc_info=exc)

    async def shutdown(self) -> None:
        coros = [p.shutdown() for p in self.providers.values()]
        await asyncio.gather(*coros, return_exceptions=True)

    def resolve(self, name: str | None) -> SearchProvider:
        """Return the named provider, or the default when name is None.

        Raises HTTPException(400) if the provider is unknown,
        or HTTPException(503) if no providers are configured.
        """
        if not self.providers:
            raise HTTPException(
                status_code=503,
                detail="Search hub has no providers configured.",
            )

        target = name or self._default
        if target is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No default search provider configured. "
                    "Specify 'provider' in the request body."
                ),
            )

        provider = self.providers.get(target)
        if provider is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown search provider: {target!r}. "
                    f"Available: {list(self.providers)}"
                ),
            )
        return provider
