from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from app.core.config import settings
from app.core.logging_config import setup_logging

if TYPE_CHECKING:
    from app.services.pricing import CostBreakdown

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
        cost: "CostBreakdown | None" = None,
        session_id: str | None = None,
        trace_id: str | None = None,
        tools_defined: list[str] | None = None,
        tool_calls: list[str] | None = None,
    ) -> None: ...

    async def score(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: str | None,
        group_id: str,
    ) -> None: ...

    def trace_search_request(
        self,
        provider: str,
        query: str,
        num_results: int,
        result_count: int,
        status_code: int,
        duration_ms: float,
        group_id: str,
        cost: float | None,
        trace_id: str | None,
        session_id: str | None,
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
    cost: "CostBreakdown | None" = None,
    session_id: str | None = None,
    trace_id: str | None = None,
    tools_defined: list[str] | None = None,
    tool_calls: list[str] | None = None,
) -> None:
    """Convenience wrapper — strips IO when configured, then delegates to the emitter."""
    emitter = get_emitter()
    if emitter is None:
        logger.debug("trace_proxy_request skipped: emitter is None")
        return
    logger.debug("trace_proxy_request", model=model, provider=provider, group_id=group_id, cost=cost)
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
            cost=cost,
            session_id=session_id,
            trace_id=trace_id,
            tools_defined=tools_defined,
            tool_calls=tool_calls,
        )
    except Exception as e:
        logger.error("Error in trace_proxy_request", exc_info=e)


async def score_trace(
    request_id: str,
    name: str,
    value: float,
    comment: str | None,
    group_id: str,
) -> str:
    """Attach a user feedback score to the trace created for request_id.

    Derives the Langfuse trace_id from request_id using the same seed formula
    as the proxy, so no state needs to be stored between request and feedback.
    Returns the derived trace_id (empty string when tracing is disabled).
    """
    emitter = get_emitter()
    if emitter is None:
        logger.debug("score_trace skipped: emitter is None")
        return ""

    from langfuse import Langfuse
    trace_id = Langfuse.create_trace_id(seed=request_id)

    logger.debug("score_trace", request_id=request_id, trace_id=trace_id, name=name, value=value, group_id=group_id)
    try:
        await emitter.score(trace_id=trace_id, name=name, value=value, comment=comment, group_id=group_id)
    except Exception as e:
        logger.error("Error in score_trace", exc_info=e)

    return trace_id


def trace_search_request(
    provider: str,
    query: str,
    num_results: int,
    result_count: int,
    status_code: int,
    duration_ms: float,
    group_id: str,
    cost: float | None = None,
    trace_id: str | None = None,
    session_id: str | None = None,
) -> None:
    """Convenience wrapper — emit a search request trace via the configured emitter."""
    emitter = get_emitter()
    if emitter is None:
        logger.debug("trace_search_request skipped: emitter is None")
        return
    if not settings.tracing_log_io:
        query = None
    logger.debug(
        "trace_search_request",
        provider=provider,
        query=query,
        result_count=result_count,
        group_id=group_id,
        cost=cost,
        session_id=session_id,
    )
    try:
        emitter.trace_search_request(
            provider=provider,
            query=query,
            num_results=num_results,
            result_count=result_count,
            status_code=status_code,
            duration_ms=duration_ms,
            group_id=group_id,
            cost=cost,
            trace_id=trace_id,
            session_id=session_id,
        )
    except Exception as e:
        logger.error("Error in trace_search_request", exc_info=e)


def shutdown() -> None:
    """Flush and shutdown the current emitter — called from lifespan."""
    global _emitter
    if _emitter is not None:
        try:
            _emitter.shutdown()
        except Exception as e:
            logger.error("Error shutting down trace emitter", exc_info=e)
        _emitter = None
