
from fastapi import FastAPI
from app.core.config import settings
from app.core.logging_config import setup_logging
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
    @app.on_event("startup")
    async def startup_event():
        logger.info("Application startup...")

    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Application shutdown...")

    return app
