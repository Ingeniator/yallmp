
from fastapi import FastAPI, Depends, Request
from app.core.config import settings
from app.core.logging_config import setup_logging
from app.middlewares.logging_middleware import LoggingMiddleware
from app.middlewares.metrics_middleware import PrometheusMiddleware, get_metrics_registry
from app.schemas.health import HealthCheck
from app.schemas.prompt import ChainMetadataForTracking, ChainType
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, Response
from contextlib import asynccontextmanager
import json
import pathlib

logger = setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup...")
    client = None
    if settings.proxy_enabled:
        from app.core.proxy import create_async_client
        client = await create_async_client()

    app.state.client = client

    # Initialize LLM Hub if enabled
    llm_hub = None
    if settings.llm_hub_enabled:
        from app.services.llm_hub import LlmHub
        llm_hub = LlmHub()
        llm_hub.load_providers()
        await llm_hub.startup()

    app.state.llm_hub = llm_hub

    # Initialize pricing cache
    pricing_cache = None
    if llm_hub and llm_hub.providers:
        from app.services.pricing import PricingCache
        pricing_cache = PricingCache(llm_hub.providers)
        await pricing_cache.startup()
    elif settings.proxy_pricing_endpoint:
        from app.services.pricing import PricingCache
        pricing_cache = PricingCache.from_endpoint(
            url=settings.proxy_pricing_endpoint,
            prefix=settings.proxy_pricing_prefix,
            currency=settings.proxy_pricing_currency,
        )
        await pricing_cache.startup()
    elif settings.proxy_pricing_config:
        from app.services.pricing import PricingCache
        pricing_cache = PricingCache.from_json(settings.proxy_pricing_config)

    app.state.pricing_cache = pricing_cache

    # Initialize billing Redis + limits
    app.state.billing_redis = None
    app.state.billing_limits = {}
    if settings.billing_enabled:
        import redis.asyncio as aioredis
        from app.services.billing import load_limits
        app.state.billing_redis = aioredis.from_url(settings.billing_redis_url, decode_responses=True)
        app.state.billing_limits = load_limits(settings.billing_limits_path)
        logger.info("Billing enabled", redis_url=settings.billing_redis_url)

    yield

    if pricing_cache:
        await pricing_cache.shutdown()

    if app.state.billing_redis:
        await app.state.billing_redis.aclose()

    from app.services.tracing import shutdown as tracing_shutdown
    tracing_shutdown()

    if llm_hub:
        await llm_hub.shutdown()
    if app.state.client:
        await app.state.client.aclose()
    logger.info("Application shutdown...")

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title=settings.app_name, debug=settings.debug, root_path=settings.root_path, lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Add Logging Middleware
    app.add_middleware(LoggingMiddleware)
    # Add Billing Middleware (limit enforcement, runs before Prometheus)
    from app.middlewares.billing_middleware import BillingMiddleware
    app.add_middleware(BillingMiddleware)
    # Add Prometheus middleware
    app.add_middleware(PrometheusMiddleware)

    # Expose metrics endpoint
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    @app.get("/metrics")
    async def get_metrics():
        return Response(content=generate_latest(get_metrics_registry()), media_type=CONTENT_TYPE_LATEST)

    @app.get("/livez")
    async def livez():
        """Liveness probe — process is alive, no dependency checks."""
        return {"status": "ok"}

    @app.get("/ready")
    async def ready():
        """Readiness probe — returns 200 if all enabled components are healthy, 503 otherwise."""
        from app.core.proxy import circuit_breaker

        components = {
            "proxy": "ok" if settings.proxy_enabled else "disabled",
            "llm_hub": "ok" if settings.llm_hub_enabled else "disabled",
            "chain_hub": "ok" if settings.chain_hub_enabled else "disabled",
            "prompt_hub": "ok" if settings.prompt_hub_enabled else "disabled",
            "dashboard": "ok" if settings.dashboard_enabled else "disabled",
        }

        if settings.proxy_enabled:
            cb_status = await circuit_breaker.get_status()
            if cb_status["circuit_open"]:
                components["proxy"] = "degraded"

        llm_hub = getattr(app.state, "llm_hub", None)
        if settings.llm_hub_enabled and llm_hub and llm_hub.providers:
            for prefix, provider in llm_hub.providers.items():
                cb = await provider.circuit_breaker.get_status()
                if cb["circuit_open"]:
                    components["llm_hub"] = "degraded"
                    break

        enabled = {k: v for k, v in components.items() if v != "disabled"}
        if all(v == "ok" for v in enabled.values()):
            return JSONResponse(status_code=200, content={"status": "ok", "components": components})
        return JSONResponse(status_code=503, content={"status": "degraded", "components": components})

    @app.get("/health")
    async def health_check() -> HealthCheck:
        """Full health status with component details — for dashboards and monitoring."""
        from app.core.proxy import circuit_breaker

        components = {
            "proxy": "ok" if settings.proxy_enabled else "disabled",
            "llm_hub": "ok" if settings.llm_hub_enabled else "disabled",
            "chain_hub": "ok" if settings.chain_hub_enabled else "disabled",
            "prompt_hub": "ok" if settings.prompt_hub_enabled else "disabled",
            "dashboard": "ok" if settings.dashboard_enabled else "disabled",
            "tracing": "ok" if settings.tracing_enabled else "disabled",
        }

        details: dict = {}

        # Check circuit breaker state (legacy single-provider proxy)
        if settings.proxy_enabled:
            cb_status = await circuit_breaker.get_status()
            if cb_status["circuit_open"]:
                components["proxy"] = "degraded"
                details["proxy"] = {"circuit_breaker": "open"}

        # Check LLM Hub providers — probe each provider's circuit breaker
        llm_hub = getattr(app.state, "llm_hub", None)
        if settings.llm_hub_enabled and llm_hub and llm_hub.providers:
            provider_statuses = {}
            for prefix, provider in llm_hub.providers.items():
                cb = await provider.circuit_breaker.get_status()
                if cb["circuit_open"]:
                    provider_statuses[prefix] = "circuit_open"
                else:
                    provider_statuses[prefix] = "ok"

            if any(v != "ok" for v in provider_statuses.values()):
                if all(v != "ok" for v in provider_statuses.values()):
                    components["llm_hub"] = "degraded"
                else:
                    components["llm_hub"] = "degraded"
                details["llm_hub_providers"] = provider_statuses

        # Check tracing backend reachability
        if settings.tracing_enabled:
            if not settings.tracing_host:
                components["tracing"] = "degraded"
                details["tracing"] = "tracing_host is not configured"
            else:
                import httpx as _httpx
                try:
                    async with _httpx.AsyncClient(timeout=3) as client:
                        resp = await client.get(f"{settings.tracing_host.rstrip('/')}/livez")
                        resp.raise_for_status()
                    components["tracing"] = "ok"
                except Exception as exc:
                    components["tracing"] = "degraded"
                    details["tracing"] = str(exc)
        else:
            components["tracing"] = "disabled"

        enabled = {k: v for k, v in components.items() if v != "disabled"}
        status = "ok" if all(v == "ok" for v in enabled.values()) else "degraded"

        return HealthCheck(
            status=status,
            components=components,
            version=settings.version,
            details=details if details else None,
        )

    if settings.dashboard_enabled:
        from app.services.dashboard import get_dashboard_json
        from app.core.security import sanitize_group_id

        _dashboard_html = pathlib.Path(__file__).resolve().parent.parent.joinpath(
            "templates", "dashboard.html"
        ).read_text()

        def _effective_identity(request: Request) -> tuple[str, str]:
            """Return (role, group_id), applying dev cookie override when enabled."""
            role = request.headers.get("x-role", "").upper()
            group_id = sanitize_group_id(request.headers.get("x-group-id", ""))
            if settings.dashboard_dev_role_switcher:
                from urllib.parse import unquote
                cookie_role = request.cookies.get("dev_role", "").upper()
                cookie_group = unquote(request.cookies.get("dev_group", ""))
                if cookie_role:
                    role = cookie_role
                if cookie_group:
                    group_id = sanitize_group_id(cookie_group)
            return role, group_id

        @app.get("/dashboard")
        async def dashboard():
            return HTMLResponse(content=_dashboard_html)

        @app.get("/dashboard/api/config")
        async def dashboard_api_config():
            return {"dev_switcher": settings.dashboard_dev_role_switcher}

        @app.get("/dashboard/api/metrics")
        async def dashboard_api_metrics(
            request: Request,
            time_window: str = "",
            start: str = "",
            end: str = "",
        ):
            role, group_id = _effective_identity(request)
            return await get_dashboard_json(
                group_id=group_id,
                is_org_admin=role == "ORG_ADMIN",
                is_super_admin=role == "SUPER_ADMIN",
                time_window=time_window, start=start, end=end,
            )

        @app.get("/dashboard/api/billing")
        async def dashboard_api_billing(request: Request):
            from app.services.billing import get_billing_summary
            role, group_id = _effective_identity(request)
            redis = getattr(request.app.state, "billing_redis", None)
            limits = getattr(request.app.state, "billing_limits", {})
            if not redis or not settings.billing_enabled:
                return {"enabled": False}
            return await get_billing_summary(redis, limits, group_id, role)

        @app.get("/dashboard/api/trends")
        async def dashboard_api_trends(
            request: Request,
            time_window: str = "7d",
            group_by: str = "group_id",
            start: str = "",
            end: str = "",
        ):
            if settings.dashboard_metrics_backend != "prometheus":
                return {"available": False, "labels": [], "series": []}
            from app.services.dashboard_prometheus import fetch_cost_trends
            role, group_id = _effective_identity(request)
            prom_auth = None
            if settings.dashboard_prometheus_user:
                prom_auth = (settings.dashboard_prometheus_user, settings.dashboard_prometheus_password)
            prom_verify = (
                settings.dashboard_prometheus_ca_bundle
                if settings.dashboard_prometheus_ca_bundle
                else settings.dashboard_prometheus_verify_ssl
            )
            return await fetch_cost_trends(
                url=settings.dashboard_prometheus_url,
                timeout=settings.dashboard_prometheus_timeout,
                group_id=group_id,
                is_org_admin=role == "ORG_ADMIN",
                is_super_admin=role == "SUPER_ADMIN",
                auth=prom_auth,
                verify=prom_verify,
                time_window=time_window,
                group_by=group_by,
                start=start,
                end=end,
            )

        @app.get("/dashboard/api/who")
        async def dashboard_api_who(request: Request):
            role, group_id = _effective_identity(request)
            if not role:
                role = "USER"
            org = group_id.split("/")[0] if group_id else ""
            return {"role": role, "group_id": group_id, "org": org}

        @app.get("/dashboard/api/sessions")
        async def dashboard_api_sessions(
            request: Request,
            start: str = "",
            end: str = "",
            limit: int = 50,
            offset: int = 0,
        ):
            if not settings.tracing_host:
                return {"sessions": []}
            import httpx as _httpx
            role, group_id = _effective_identity(request)
            params = {"limit": limit, "offset": offset}
            if start:
                params["start"] = start
            if end:
                params["end"] = end
            try:
                async with _httpx.AsyncClient(timeout=30) as client:
                    resp = await client.get(
                        f"{settings.tracing_host.rstrip('/')}/api/public/sessions",
                        params=params,
                        headers={"x-group-id": group_id, "x-role": role},
                    )
                    resp.raise_for_status()
                    return resp.json()
            except Exception as exc:
                return {"sessions": [], "error": str(exc)}

        @app.get("/dashboard/api/sessions/{session_id}")
        async def dashboard_api_session_traces(session_id: str, request: Request):
            if not settings.tracing_host:
                return {"session_id": session_id, "traces": []}
            import httpx as _httpx
            role, group_id = _effective_identity(request)
            try:
                async with _httpx.AsyncClient(timeout=30) as client:
                    resp = await client.get(
                        f"{settings.tracing_host.rstrip('/')}/api/public/sessions/{session_id}",
                        headers={"x-group-id": group_id, "x-role": role},
                    )
                    resp.raise_for_status()
                    return resp.json()
            except Exception as exc:
                return {"session_id": session_id, "traces": [], "error": str(exc)}

    if settings.proxy_enabled:
        from app.core.proxy import proxy_request_with_retries, get_model_version, proxy_request_to_provider
        from app.services.llm_authentication import get_authorization_headers

        async def add_authz_header():
            return await get_authorization_headers(app.state.client)

        @app.api_route("/llm/version", methods=["GET"])
        async def get_llm_version(request: Request, custom_headers: dict[str, str] = Depends(add_authz_header), model_name: str = "GigaChat"):
            return await get_model_version(model_name, app.state.client, request, custom_headers)

        # Models endpoint — must be registered BEFORE the catch-all /llm/{path}
        @app.get("/llm/v1/models")
        async def get_models(request: Request, custom_headers: dict[str, str] = Depends(add_authz_header)):
            llm_hub = app.state.llm_hub
            if llm_hub and llm_hub.providers:
                return JSONResponse(content=llm_hub.get_merged_models())
            # Fallback: proxy to upstream
            return await proxy_request_with_retries(app.state.client, "v1/models", request, custom_headers)

        @app.api_route("/llm/{full_path:path}", methods=["GET", "POST"])
        async def proxy_request(full_path: str, request: Request, custom_headers: dict[str, str] = Depends(add_authz_header)):
            llm_hub = app.state.llm_hub

            # Multi-provider routing for POST requests when hub is enabled
            if request.method == "POST" and llm_hub and llm_hub.providers:
                body = await request.body()
                try:
                    body_json = json.loads(body)
                    model = body_json.get("model", "")
                except (json.JSONDecodeError, AttributeError):
                    model = ""

                if model:
                    resolved = llm_hub.resolve_model(model)
                    if resolved:
                        provider, stripped_model = resolved
                        provider_headers = await provider.get_auth_headers()
                        return await proxy_request_to_provider(
                            provider=provider,
                            path=full_path,
                            request=request,
                            auth_headers=provider_headers,
                            original_model=model,
                            stripped_model=stripped_model,
                            pricing_cache=app.state.pricing_cache,
                        )

            # Legacy single-provider path
            return await proxy_request_with_retries(app.state.client, full_path, request, custom_headers, pricing_cache=app.state.pricing_cache)

    if settings.prompt_hub_enabled:
        from app.services.prompt_manager import get_prompt_store
        from app.schemas.prompt import PromptVariables as _PromptVars

        @app.get("/prompts")
        async def get_prompts(category: str | None = None):
            return await get_prompt_store().get_prompts(category)

        @app.post("/prompt/format/{name}")
        async def format_prompt(name: str, data: _PromptVars):
            return await get_prompt_store().format_prompt(name, data)

    if settings.chain_hub_enabled:
        from app.services.chain_manager import get_chain_store
        from app.schemas.prompt import PromptVariables

        @app.get("/chains")
        async def get_chains(category: str | None = None):
            return await get_chain_store().get_chains(category)

        @app.post("/chain/execute/{name}")
        async def chain_execute(request: Request, name: str, data: PromptVariables, model_name: str = None):
            metadata = ChainMetadataForTracking(chain_type=ChainType.chain, chain_name = name, group_id = request.headers.get("x-group-id", "unknown"))
            return await get_chain_store().execute(name, data, model_name, metadata)

    if settings.prompt_hub_enabled and settings.chain_hub_enabled:
        @app.post("/prompt/execute/{name}")
        async def execute_prompt(request: Request, name: str, data: PromptVariables, model_name: str = None):
            prompt = await get_prompt_store().format_prompt(name, data)
            metadata = ChainMetadataForTracking(chain_type=ChainType.prompt, chain_name = name, group_id = request.headers.get("x-group-id", "unknown"))
            return await get_chain_store().execute_prompt(prompt=prompt, model_name=model_name, metadata=metadata)

    return app
