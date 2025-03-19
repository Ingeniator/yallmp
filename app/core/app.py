
from fastapi import FastAPI, Depends, Request
from app.core.config import settings
from app.core.logging_config import setup_logging
from app.middlewares.logging_middleware import LoggingMiddleware
from app.middlewares.metrics_middleware import PrometheusMiddleware, metrics
from app.schemas.health import HealthCheck

logger = setup_logging()

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title=settings.app_name, debug=settings.debug)

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
            "proxy": "ok" if settings.raw_proxy_llm_enabled else "disabled",
            "chain_hub": "ok" if settings.chain_hub_enabled else "disabled",
            "prompt_hub": "ok" if settings.prompt_hub_enabled else "disabled"
        }
        
        status = "ok" if all(v == "ok" for v in components.values()) else "degraded"
        
        return HealthCheck(
            status=status,
            components=components,
            version=settings.version
        )

    if settings.raw_proxy_llm_enabled:
        from app.core.proxy import proxy_request_with_retries
        from app.services.llm_authentication import get_authorization_headers
        @app.api_route("/proxy/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
        async def proxy_request(full_path: str, request: Request, custom_headers: dict[str, str] = Depends(get_authorization_headers)):
            return await proxy_request_with_retries(full_path, request, custom_headers)

    if settings.prompt_hub_enabled:
        from app.services.prompt_manager import promptStore, PromptVariables
        @app.get("/prompts")
        async def get_prompts():
            return await promptStore.get_prompts()

        @app.post("/prompt/{name}")
        async def format_prompt(name: str, data: PromptVariables):
            return await promptStore.format_prompt(name, data)

    if settings.chain_hub_enabled:
        from app.services.chain_manager import chainStore, PromptVariables
        @app.get("/chains")
        async def get_chains():
            return await chainStore.get_chains()

        @app.post("/chain/{name}")
        async def chain_execute(name: str, data: PromptVariables):
            return await chainStore.execute(name, data)

    @app.on_event("startup")
    async def startup_event():
        logger.info("Application startup...")

    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Application shutdown...")

    return app
