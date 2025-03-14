
from fastapi import Depends, FastAPI, Request
from app.core.config import settings
from app.core.logging_config import setup_logging
from app.services.authenticated_llm import get_authenticated_model
from app.middlewares.logging_middleware import LoggingMiddleware
from app.middlewares.metrics_middleware import PrometheusMiddleware, metrics
from langchain.chat_models.base import BaseChatModel

logger = setup_logging()

def create_fake_app() -> FastAPI:
    """Create and configure a FastAPI application for testing."""
    fake_app = FastAPI(title="Fake LLM and AuthPoint", debug=settings.debug)
    
    @fake_app.get("/{path:path}")
    async def chat(path: str, llm: BaseChatModel = Depends(get_authenticated_model)):
        return llm.invoke("Hello, world!")

    return fake_app

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title=settings.app_name, debug=settings.debug)

    # Add Logging Middleware
    app.add_middleware(LoggingMiddleware)
    # Add Prometheus middleware
    app.add_middleware(PrometheusMiddleware)

    # Mount the sub-app under `/fake`
    app.mount("/fake", create_fake_app())

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
