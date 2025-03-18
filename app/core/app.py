
from fastapi import FastAPI, Depends, Request
from app.core.config import settings
from app.core.proxy import proxy_request_with_retries
from app.core.logging_config import setup_logging
from app.services.llm_authentication import get_authorization_headers
from app.middlewares.logging_middleware import LoggingMiddleware
from app.middlewares.metrics_middleware import PrometheusMiddleware, metrics

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

    @app.get("/status")
    async def status():
        return {"status": "Application is running"}

    @app.api_route("/llm/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
    async def proxy_request(full_path: str, request: Request, custom_headers: dict[str, str] = Depends(get_authorization_headers)):
        return await proxy_request_with_retries(full_path, request, custom_headers)

    @app.on_event("startup")
    async def startup_event():
        logger.info("Application startup...")

    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Application shutdown...")

    return app
