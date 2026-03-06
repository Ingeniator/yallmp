from __future__ import annotations

from app.core.logging_config import setup_logging

logger = setup_logging()


class LangfuseEmitter:
    """TraceEmitter implementation backed by Langfuse.

    Per-group isolation: each unique group_id gets its own Langfuse client
    with public_key=group_id, so the backend can separate logs by group.
    """

    def __init__(self):
        from langfuse import Langfuse

        self._default_client = Langfuse()
        self._clients: dict[str, object] = {}
        logger.info("Langfuse client initialized")

    def _get_client(self, group_id: str):
        if not group_id or group_id == "unknown":
            return self._default_client
        if group_id not in self._clients:
            from langfuse import Langfuse

            self._clients[group_id] = Langfuse(public_key=group_id, secret_key=group_id)
        return self._clients[group_id]

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
        client = self._get_client(group_id)
        trace = client.trace(
            name="llm-proxy",
            user_id=group_id,
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

            group_id = metadata.get("group_id", "")
            client = self._get_client(group_id)
            return CallbackHandler(
                trace_name=trace_name,
                metadata=metadata,
                langfuse_client=client,
            )
        except Exception as e:
            logger.error("Failed to create Langfuse callback handler", exc_info=e)
            return None

    def shutdown(self) -> None:
        self._default_client.flush()
        self._default_client.shutdown()
        for client in self._clients.values():
            client.flush()
            client.shutdown()
        self._clients.clear()
