from __future__ import annotations

from app.core.config import settings
from app.core.logging_config import setup_logging

logger = setup_logging()

_client = None


def get_client():
    """Return shared Langfuse client, creating it on first call. Returns None if disabled."""
    global _client
    if not settings.langfuse_enabled:
        return None
    if _client is None:
        from langfuse import Langfuse
        _client = Langfuse()
        logger.info("Langfuse client initialized")
    return _client


def shutdown():
    """Flush and shutdown — called from lifespan."""
    global _client
    if _client is not None:
        try:
            _client.flush()
            _client.shutdown()
        except Exception as e:
            logger.error("Error shutting down Langfuse client", exc_info=e)
        _client = None


def trace_proxy_request(
    model: str,
    provider: str | None,
    input_body: dict | None,
    output_body: dict | None,
    status_code: int,
    usage: dict | None,
    duration_ms: float,
    group_id: str,
    is_streaming: bool,
):
    """Create a Langfuse trace + generation for a proxy LLM call."""
    client = get_client()
    if client is None:
        return

    if not settings.langfuse_log_io:
        input_body = None
        output_body = None

    try:
        trace = client.trace(
            name="llm-proxy",
            metadata={
                "provider": provider,
                "group_id": group_id,
                "is_streaming": is_streaming,
                "status_code": status_code,
            },
        )
        trace.generation(
            name="llm-call",
            model=model,
            input=input_body,
            output=output_body,
            usage=usage,
            metadata={
                "duration_ms": duration_ms,
                "status_code": status_code,
                "provider": provider,
            },
        )
    except Exception as e:
        logger.error("Error creating Langfuse trace", exc_info=e)
