from __future__ import annotations

from app.core.config import settings
from app.core.logging_config import setup_logging

logger = setup_logging()


class LangfuseEmitter:
    """TraceEmitter implementation backed by Langfuse v3 SDK.

    Uses the native OTEL-based Langfuse client which sends spans
    to /api/public/otel/v1/traces via protobuf.

    Per-group isolation: each unique group_id gets its own Langfuse client
    with public_key=group_id, so the backend can separate logs by group.
    """

    def __init__(self):
        from langfuse import Langfuse

        self._host = settings.tracing_host
        self._public_key = settings.tracing_public_key
        self._secret_key = settings.tracing_secret_key
        self._default_client = Langfuse(
            host=self._host,
            public_key=self._public_key,
            secret_key=self._secret_key,
        )
        self._clients: dict[str, object] = {}

        from importlib.metadata import version as pkg_version
        langfuse_ver = pkg_version("langfuse")
        logger.info("Langfuse client initialized (OTEL mode)", host=self._host, langfuse_version=langfuse_ver)

    def _get_client(self, group_id: str):
        if not group_id or group_id == "unknown":
            return self._default_client
        if group_id not in self._clients:
            from langfuse import Langfuse

            self._clients[group_id] = Langfuse(
                host=self._host,
                public_key=group_id,
                secret_key=group_id,
            )
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
        cost: float | None = None,
    ) -> None:
        client = self._get_client(group_id)

        usage_details = {}
        if usage:
            if "prompt_tokens" in usage:
                usage_details["input"] = usage["prompt_tokens"]
            if "completion_tokens" in usage:
                usage_details["output"] = usage["completion_tokens"]
            if "total_tokens" in usage:
                usage_details["total"] = usage["total_tokens"]

        metadata = {
            "provider": provider,
            "group_id": group_id,
            "is_streaming": is_streaming,
            "status_code": status_code,
            "duration_ms": duration_ms,
        }

        cost_details = None
        if cost is not None:
            cost_details = {"total": cost}
            metadata["cost"] = cost

        span = client.start_span(name="llm-proxy", metadata=metadata)
        span.start_generation(
            name="llm-proxy",
            model=model,
            input=input_body,
            output=output_body,
            usage_details=usage_details or None,
            cost_details=cost_details,
        )
        span.end()

    def get_langchain_callback(self, trace_name: str, metadata: dict) -> object | None:
        try:
            from langfuse.langchain import CallbackHandler

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
