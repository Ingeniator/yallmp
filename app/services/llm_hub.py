import os
import json
import asyncio
from pathlib import Path

from httpx import AsyncClient, Timeout, Limits

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.core.proxy import CircuitBreaker
from app.schemas.provider import LlmProviderConfig, AuthType
from app.services.token_manager import OIDCTokenManager

logger = setup_logging()


class LlmProvider:
    """Holds runtime state for a single LLM provider."""

    def __init__(self, config: LlmProviderConfig):
        self.config = config
        self.client: AsyncClient | None = None
        self.circuit_breaker = CircuitBreaker()
        self.token_manager: OIDCTokenManager | None = None

    async def startup(self):
        cert = None
        if self.config.auth.type == AuthType.CERT:
            cert = (self.config.auth.cert_path, self.config.auth.cert_key_path)

        t = self.config.timeout
        self.client = AsyncClient(
            cert=cert,
            timeout=Timeout(connect=t.connect, read=t.read, write=t.write, pool=t.pool),
            limits=Limits(
                max_connections=settings.max_connections,
                max_keepalive_connections=settings.max_keepalive_connections,
            ),
            verify=self.config.verify_ssl,
        )

        if self.config.auth.type == AuthType.BEARER:
            self.token_manager = OIDCTokenManager(
                authorization_url=self.config.auth.oidc_url,
                credentials=self.config.auth.credentials,
                scope=self.config.auth.scope,
            )

    async def shutdown(self):
        if self.client:
            await self.client.aclose()

    async def get_auth_headers(self) -> dict[str, str]:
        auth = self.config.auth
        if auth.type == AuthType.BEARER and self.token_manager and self.client:
            token = await self.token_manager.get_token(self.client)
            return {"Authorization": f"Bearer {token}"}
        if auth.type == AuthType.APIKEY and auth.api_key:
            return {"X-API-KEY": auth.api_key}
        return {}


class LlmHub:
    """Registry of LLM providers loaded from JSON config files."""

    def __init__(self):
        self.providers: dict[str, LlmProvider] = {}

    def load_providers(self, directory: str | None = None):
        directory = directory or settings.llm_hub_directory
        hub_path = Path(directory)
        if not hub_path.is_dir():
            logger.warning(f"LLM hub directory not found: {hub_path}")
            return

        seen_prefixes: set[str] = set()
        for json_file in sorted(hub_path.glob("*.json")):
            try:
                raw = json_file.read_text(encoding="utf-8")
                expanded = os.path.expandvars(raw)
                data = json.loads(expanded)

                # Skip files that don't have the new provider format (e.g. langchain configs)
                if "prefix" not in data:
                    logger.debug(f"Skipping {json_file.name}: no 'prefix' field")
                    continue

                config = LlmProviderConfig(**data)

                if config.prefix in seen_prefixes:
                    logger.error(f"Duplicate prefix '{config.prefix}' in {json_file.name}, skipping")
                    continue

                seen_prefixes.add(config.prefix)
                self.providers[config.prefix] = LlmProvider(config)
                logger.info(f"Loaded provider '{config.prefix}' from {json_file.name} with {len(config.models)} models")

            except Exception as e:
                logger.error(f"Failed to load provider from {json_file.name}: {e}")

    async def startup(self):
        for prefix, provider in self.providers.items():
            try:
                await provider.startup()
                logger.info(f"Provider '{prefix}' started")
            except Exception as e:
                logger.error(f"Failed to start provider '{prefix}': {e}")

    async def shutdown(self):
        coros = [provider.shutdown() for provider in self.providers.values()]
        await asyncio.gather(*coros, return_exceptions=True)

    def resolve_model(self, prefixed_name: str) -> tuple[LlmProvider, str] | None:
        """Split 'prefix/model' and return (provider, stripped_model) or None."""
        if "/" not in prefixed_name:
            return None

        prefix, model = prefixed_name.split("/", 1)
        provider = self.providers.get(prefix)
        if provider is None:
            # prefix doesn't match any known provider — could be a model name with /
            return None
        return provider, model

    def get_merged_models(self) -> dict:
        """Return an OpenAI-compatible model list merged from all providers."""
        data = []
        for prefix, provider in self.providers.items():
            for model in provider.config.models:
                data.append({
                    "id": f"{prefix}/{model}",
                    "object": "model",
                    "owned_by": prefix,
                    "created": 0,
                })
        return {"object": "list", "data": data}
