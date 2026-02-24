
from fastapi import FastAPI, Depends, Request
from app.core.config import settings
from app.core.logging_config import setup_logging
from app.middlewares.logging_middleware import LoggingMiddleware
from app.middlewares.metrics_middleware import PrometheusMiddleware, metrics
from app.schemas.health import HealthCheck
from app.schemas.prompt import ChainMetadataForTracking, ChainType
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import json

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

    yield

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
    # Add Prometheus middleware
    app.add_middleware(PrometheusMiddleware)

    # Expose metrics endpoint
    @app.get("/metrics")
    async def get_metrics():
        return await metrics()

    @app.get("/health")
    async def health_check() -> HealthCheck:
        """Check the health of the service and its components."""
        components = {
            "proxy": "ok" if settings.proxy_enabled else "disabled",
            "chain_hub": "ok" if settings.chain_hub_enabled else "disabled",
            "prompt_hub": "ok" if settings.prompt_hub_enabled else "disabled"
        }

        enabled = {k: v for k, v in components.items() if v != "disabled"}
        status = "ok" if all(v == "ok" for v in enabled.values()) else "degraded"

        return HealthCheck(
            status=status,
            components=components,
            version=settings.version
        )

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
                        )

            # Legacy single-provider path
            return await proxy_request_with_retries(app.state.client, full_path, request, custom_headers)

    if settings.prompt_hub_enabled:
        from app.services.prompt_manager import promptStore, PromptVariables
        @app.get("/prompts")
        async def get_prompts(category: str | None = None):
            return await promptStore.get_prompts(category)

        @app.post("/prompt/format/{name}")
        async def format_prompt(name: str, data: PromptVariables):
            return await promptStore.format_prompt(name, data)

    if settings.chain_hub_enabled:
        from app.services.chain_manager import chainStore, PromptVariables
        @app.get("/chains")
        async def get_chains(category: str | None = None):
            return await chainStore.get_chains(category)

        @app.post("/chain/execute/{name}")
        async def chain_execute(request: Request, name: str, data: PromptVariables, model_name: str = None):
            metadata = ChainMetadataForTracking(chain_type=ChainType.chain, chain_name = name, group_id = request.headers.get("x-group-id", "unknown"))
            return await chainStore.execute(name, data, model_name, metadata)

    if settings.prompt_hub_enabled and settings.chain_hub_enabled:
        @app.post("/prompt/execute/{name}")
        async def execute_prompt(request: Request, name: str, data: PromptVariables, model_name: str = None):
            prompt = await promptStore.format_prompt(name, data)
            metadata = ChainMetadataForTracking(chain_type=ChainType.prompt, chain_name = name, group_id = request.headers.get("x-group-id", "unknown"))
            return await chainStore.execute_prompt(prompt=prompt, model_name=model_name, metadata=metadata)

    return app
