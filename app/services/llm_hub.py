import os
import json
import hashlib
import re
import time
import asyncio
from collections.abc import Mapping
from pathlib import Path

from httpx import AsyncClient, Timeout, Limits

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.core.proxy import CircuitBreaker
from app.core.security import sanitize_group_id
from app.schemas.provider import (
    LlmProviderConfig,
    AliasEntry,
    AuthType,
    HEADER_PLACEHOLDER,
)
from app.services.token_manager import OIDCTokenManager

logger = setup_logging()

_MODELS_TTL = 3600  # seconds a fetched model list is cached
_MODELS_RETRY_AFTER = 60  # seconds before retrying a failed model list fetch
_MODELS_FETCH_TIMEOUT = 10  # seconds; independent of the provider's completion timeouts
_MODELS_PRUNE_AFTER = _MODELS_TTL  # seconds past expiry before an entry is dropped
_LEGACY_MODELS_KEY = "__legacy__"
# ${VAR} left over after os.path.expandvars, i.e. an unset environment variable.
# ${headers.X} are PASSTHROUGH placeholders, resolved per request.
_UNRESOLVED_ENV_VAR = re.compile(r"\$\{(?!headers\.)([^}]+)\}")


async def _fetch_model_list(
    client: AsyncClient, url: str, headers: Mapping[str, str]
) -> list[dict]:
    """GET an OpenAI-style model list and return its ``data`` entries."""
    resp = await client.get(url, headers=dict(headers), timeout=_MODELS_FETCH_TIMEOUT)
    resp.raise_for_status()
    body = resp.json()
    entries = body.get("data") if isinstance(body, dict) else None
    if not isinstance(entries, list):
        raise ValueError("response has no 'data' list")
    return [e for e in entries if isinstance(e, dict) and isinstance(e.get("id"), str)]


def render_passthrough_headers(
    templates: dict[str, str], request_headers: Mapping[str, str]
) -> dict[str, str]:
    """Build upstream headers from templates like "Bearer ${headers.X-API-KEY}".

    A header is omitted if any placeholder it references is absent from the request.
    """
    result: dict[str, str] = {}
    for name, template in templates.items():
        missing = False

        def _sub(m: re.Match) -> str:
            nonlocal missing
            value = request_headers.get(m.group(1))
            if not value:
                missing = True
                return ""
            return value

        rendered = HEADER_PLACEHOLDER.sub(_sub, template)
        if not missing:
            result[name] = rendered
    return result


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

        if self.config.auth.type == AuthType.OIDC:
            self.token_manager = OIDCTokenManager(
                authorization_url=self.config.auth.oidc_url,
                credentials=self.config.auth.credentials,
                scope=self.config.auth.scope,
            )

    async def shutdown(self):
        if self.client:
            await self.client.aclose()

    async def get_auth_headers(
        self, request_headers: Mapping[str, str] | None = None
    ) -> dict[str, str]:
        auth = self.config.auth
        if auth.type == AuthType.PASSTHROUGH:
            return render_passthrough_headers(
                auth.passthrough_templates(), request_headers or {}
            )
        if auth.type == AuthType.OIDC and self.token_manager and self.client:
            token = await self.token_manager.get_token(self.client)
            return {"Authorization": f"Bearer {token}"}
        if auth.type == AuthType.BEARER and auth.api_key:
            return {"Authorization": f"Bearer {auth.api_key}"}
        if auth.type == AuthType.APIKEY and auth.api_key:
            return {"X-API-KEY": auth.api_key}
        return {}


class LlmHub:
    """Registry of LLM providers loaded from JSON config files."""

    def __init__(self):
        self.providers: dict[str, LlmProvider] = {}
        self.aliases: dict[str, AliasEntry] = {}
        # key -> (expires_at monotonic, entries)
        self._models_cache: dict[str, tuple[float, list[dict]]] = {}
        self._models_locks: dict[str, asyncio.Lock] = {}

    def load_providers(self, directory: str | None = None):
        directory = directory or settings.llm_hub_directory
        hub_path = Path(directory)
        if not hub_path.is_dir():
            logger.warning(f"LLM hub directory not found: {hub_path}")
            return

        seen_prefixes: set[str] = set()
        files = sorted(hub_path.glob("*.json"))
        for json_file in files:
            try:
                raw = json_file.read_text(encoding="utf-8")
                expanded = os.path.expandvars(raw)
                data = json.loads(expanded)

                # Skip files that don't have the new provider format (e.g. langchain configs)
                if "prefix" not in data:
                    logger.debug(f"Skipping {json_file.name}: no 'prefix' field")
                    continue

                unresolved = sorted(set(_UNRESOLVED_ENV_VAR.findall(expanded)))
                if unresolved:
                    logger.error(
                        f"Skipping {json_file.name}: environment variables not set: "
                        f"{', '.join(unresolved)}"
                    )
                    continue

                config = LlmProviderConfig(**data)

                if config.prefix in seen_prefixes:
                    logger.error(
                        f"Duplicate prefix '{config.prefix}' in {json_file.name}, skipping"
                    )
                    continue

                seen_prefixes.add(config.prefix)
                self.providers[config.prefix] = LlmProvider(config)
                logger.info(
                    f"Loaded provider '{config.prefix}' from {json_file.name} with {len(config.models)} models"
                )

            except Exception as e:
                logger.error(f"Failed to load provider from {json_file.name}: {e}")

        self._load_aliases(hub_path)

    def load_aliases(self, directory: str | None = None):
        """Load aliases.json from *directory* (defaults to llm_hub_directory).

        Safe to call without providers — used by the single-provider path.
        """
        hub_path = Path(directory or settings.llm_hub_directory)
        self._load_aliases(hub_path)

    def _load_aliases(self, hub_path: Path):
        aliases_file = hub_path / "aliases.json"
        if not aliases_file.exists():
            return
        try:
            data = json.loads(aliases_file.read_text(encoding="utf-8"))
            for name, entry in data.items():
                self.aliases[name] = AliasEntry(**entry)
            logger.info(
                f"Loaded {len(self.aliases)} model aliases from {aliases_file.name}"
            )
        except Exception as e:
            logger.error(f"Failed to load aliases from {aliases_file}: {e}")

    def resolve_alias(self, model: str) -> AliasEntry | None:
        return self.aliases.get(model)

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

    async def get_merged_models(
        self,
        legacy_client: AsyncClient | None = None,
        legacy_headers: Mapping[str, str] | None = None,
        request_headers: Mapping[str, str] | None = None,
    ) -> dict:
        """Return an OpenAI-compatible model list merged from all providers.

        Providers with a configured ``models`` list contribute those entries.
        Providers without one are queried at ``{base_url}/v1/models``; their
        entries keep the original attributes with ``id`` prefixed.
        If *legacy_client* is given, the legacy upstream's models are appended as is.
        Remote lists are cached for ``_MODELS_TTL`` seconds.
        """
        self._prune_models_cache()
        data: list[dict] = []
        tasks = []
        group_id = sanitize_group_id((request_headers or {}).get("x-group-id"))
        for prefix, provider in self.providers.items():
            if not provider.config.is_group_allowed(group_id):
                continue
            if provider.config.models:
                for model in provider.config.models:
                    data.append(
                        {
                            "id": f"{prefix}/{model}",
                            "object": "model",
                            "owned_by": prefix,
                            "created": 0,
                        }
                    )
            elif provider.client is not None:
                tasks.append(self._provider_models(prefix, provider, request_headers))

        if legacy_client is not None:

            async def legacy_fetch() -> list[dict]:
                return await _fetch_model_list(
                    legacy_client,
                    f"{settings.proxy_target_url.rstrip('/')}/v1/models",
                    legacy_headers or {},
                )

            tasks.append(self._cached_models(_LEGACY_MODELS_KEY, legacy_fetch))

        seen = {m["id"] for m in data}
        for entries in await asyncio.gather(*tasks):
            for entry in entries:
                if entry["id"] not in seen:
                    seen.add(entry["id"])
                    data.append(entry)
        return {"object": "list", "data": data}

    async def _provider_models(
        self,
        prefix: str,
        provider: LlmProvider,
        request_headers: Mapping[str, str] | None,
    ) -> list[dict]:
        key = prefix
        auth = provider.config.auth
        passthrough_headers: dict[str, str] | None = None
        if auth.type == AuthType.PASSTHROUGH:
            # The list depends on the caller's credentials: skip callers that lack
            # them and cache per credential so one caller's list never leaks to another.
            passthrough_headers = await provider.get_auth_headers(request_headers)
            if len(passthrough_headers) < len(auth.passthrough_templates()):
                return []
            digest = hashlib.sha256(
                json.dumps(sorted(passthrough_headers.items())).encode()
            ).hexdigest()[:16]
            key = f"{prefix}#{digest}"

        async def fetch() -> list[dict]:
            if passthrough_headers is not None:
                headers = passthrough_headers
            else:
                headers = await provider.get_auth_headers(request_headers)
            return await _fetch_model_list(
                provider.client,
                f"{provider.config.base_url.rstrip('/')}/v1/models",
                headers,
            )

        entries = await self._cached_models(key, fetch)
        return [{**e, "id": f"{prefix}/{e['id']}"} for e in entries]

    def _prune_models_cache(self) -> None:
        """Drop entries expired for over ``_MODELS_PRUNE_AFTER`` seconds.

        Keeps the cache from growing without bound with per-credential PASSTHROUGH
        keys. Recently expired entries stay as the stale fallback for a failed refresh.
        """
        cutoff = time.monotonic() - _MODELS_PRUNE_AFTER
        for key in [k for k, (exp, _) in self._models_cache.items() if exp < cutoff]:
            lock = self._models_locks.get(key)
            if lock is not None and lock.locked():
                continue
            del self._models_cache[key]
            self._models_locks.pop(key, None)

    async def _cached_models(self, key: str, fetch) -> list[dict]:
        """Return cached entries for *key*, refreshing via *fetch* when expired.

        While a refresh is running, other callers get the stale entries instead of
        waiting. On failure the previous entries (or []) are kept and the refresh
        is retried after ``_MODELS_RETRY_AFTER`` seconds.
        """
        cached = self._models_cache.get(key)
        if cached and cached[0] > time.monotonic():
            return cached[1]

        lock = self._models_locks.setdefault(key, asyncio.Lock())
        if cached and lock.locked():
            return cached[1]

        async with lock:
            cached = self._models_cache.get(key)
            if cached and cached[0] > time.monotonic():
                return cached[1]
            try:
                entries = await fetch()
                self._models_cache[key] = (time.monotonic() + _MODELS_TTL, entries)
                logger.info(f"Fetched {len(entries)} models from '{key}'")
            except Exception as e:
                entries = cached[1] if cached else []
                self._models_cache[key] = (
                    time.monotonic() + _MODELS_RETRY_AFTER,
                    entries,
                )
                logger.warning(
                    f"Failed to fetch models from '{key}': {type(e).__name__}: {e}; "
                    f"serving {len(entries)} cached entries"
                )
            return entries
