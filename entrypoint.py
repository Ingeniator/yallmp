"""Main application entrypoint.

Configures logging, creates FastAPI app, and starts the server.
"""

from app.core.config import settings
from app.core.app import create_app
from app.core.logging_config import setup_logging
from dotenv import load_dotenv
from pathlib import Path
# Configure logging
logger = setup_logging().bind(module=__name__)

# Create FastAPI application
app = create_app()

# load envs
env_path = Path(".env")
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting server on {settings.host}:{settings.port}")
    uvicorn.run(
        "entrypoint:app",
        workers=settings.workers,
        host=settings.host,
        port=settings.port,
        timeout_keep_alive=settings.timeout_keep_alive,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )
