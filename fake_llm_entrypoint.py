"""Fake LLM application entrypoint.

Configures logging, creates FastAPI app, and starts the server.
"""

from app.mock.fake_llm_server import create_app
from app.core.config import settings
from app.core.logging_config import setup_logging

# Configure logging
logger = setup_logging().bind(module=__name__)

# Create FastAPI application
app = create_app()

if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting server on {settings.fake_llm_host}:{settings.fake_llm_port}")
    uvicorn.run(
        "fake_llm_entrypoint:app",
        host=settings.fake_llm_host,
        port=settings.fake_llm_port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )
