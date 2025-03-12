"""Main application entrypoint.

Configures logging, creates FastAPI app, and starts the server.
"""

from app.core.app import create_app
from app.core.config import settings
from app.core.logging_config import setup_logging

# Configure logging
logger = setup_logging().bind(module=__name__)

# Create FastAPI application
app = create_app()

if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting server on {settings.host}:{settings.port}")
    uvicorn.run(
        "entrypoint:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )
