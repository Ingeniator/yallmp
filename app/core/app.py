
from fastapi import FastAPI
from app.core.config import settings
from app.core.logging_config import setup_logging

logger = setup_logging()

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title=settings.app_name, debug=settings.debug)

    @app.on_event("startup")
    async def startup_event():
        logger.info("Application startup...")

    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Application shutdown...")

    return app
