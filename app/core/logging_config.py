from app.core.config import settings
import logging
import structlog

_configured = False


def setup_logging():
    """Configures logging for the entire application.

    Safe to call multiple times — configuration runs only once.
    """
    global _configured
    if not _configured:
        _configured = True

        logging.basicConfig(
            level=settings.log_level,
            format="%(message)s",
            handlers=[logging.StreamHandler()],
        )

        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.add_log_level,
                structlog.stdlib.add_logger_name,
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(),
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy").setLevel(logging.WARNING)

    return structlog.get_logger()
