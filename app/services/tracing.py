from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.core.config import settings
from app.core.logging_config import setup_logging

logger = setup_logging()

_emitter = None


@runtime_checkable
class TraceEmitter(Protocol):
    def trace_proxy_request(
        self,
        model: str,
        provider: str | None,
        input_body: dict | None,
        output_body: dict | None,
        status_code: int,
        usage: dict | None,
        duration_ms: float,
        group_id: str,
        is_streaming: bool,
    ) -> None: ...

    def get_langchain_callback(self, trace_name: str, metadata: dict) -> object | None: ...

    def shutdown(self) -> None: ...


def get_emitter() -> TraceEmitter | None:
    """Lazy factory — returns the configured TraceEmitter or None when tracing is disabled."""
    global _emitter
    if not settings.tracing_enabled:
        return None
    if _emitter is not None:
        return _emitter
    if settings.tracing_backend == "langfuse":
        from app.services.langfuse_tracing import LangfuseEmitter

        _emitter = LangfuseEmitter()
    else:
        logger.error(f"Unknown tracing backend: {settings.tracing_backend}")
        return None
    return _emitter


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
) -> None:
    """Convenience wrapper — strips IO when configured, then delegates to the emitter."""
    emitter = get_emitter()
    if emitter is None:
        return
    if not settings.tracing_log_io:
        input_body = None
        output_body = None
    try:
        emitter.trace_proxy_request(
            model=model,
            provider=provider,
            input_body=input_body,
            output_body=output_body,
            status_code=status_code,
            usage=usage,
            duration_ms=duration_ms,
            group_id=group_id,
            is_streaming=is_streaming,
        )
    except Exception as e:
        logger.error("Error in trace_proxy_request", exc_info=e)


def shutdown() -> None:
    """Flush and shutdown the current emitter — called from lifespan."""
    global _emitter
    if _emitter is not None:
        try:
            _emitter.shutdown()
        except Exception as e:
            logger.error("Error shutting down trace emitter", exc_info=e)
        _emitter = None
