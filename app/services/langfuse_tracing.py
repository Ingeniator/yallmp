from __future__ import annotations

from app.core.logging_config import setup_logging

logger = setup_logging()


class LangfuseEmitter:
    """TraceEmitter implementation backed by Langfuse."""

    def __init__(self):
        from langfuse import Langfuse

        self._client = Langfuse()
        logger.info("Langfuse client initialized")

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
    ) -> None:
        trace = self._client.trace(
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

    def get_langchain_callback(self, trace_name: str, metadata: dict) -> object | None:
        try:
            from langfuse.callback import CallbackHandler

            return CallbackHandler(trace_name=trace_name, metadata=metadata)
        except Exception as e:
            logger.error("Failed to create Langfuse callback handler", exc_info=e)
            return None

    def shutdown(self) -> None:
        self._client.flush()
        self._client.shutdown()
