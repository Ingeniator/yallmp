from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timedelta, timezone

from langfuse import Langfuse, propagate_attributes

from app.core.config import settings
from app.core.logging_config import setup_logging

logger = setup_logging()

MAX_CLIENT_CACHE_SIZE = 128


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
        self._clients: OrderedDict[str, object] = OrderedDict()

        from importlib.metadata import version as pkg_version
        langfuse_ver = pkg_version("langfuse")
        logger.info("Langfuse client initialized (OTEL mode)", host=self._host, langfuse_version=langfuse_ver)

    def _get_client(self, group_id: str):
        if not group_id or group_id == "unknown":
            return self._default_client
        if group_id in self._clients:
            self._clients.move_to_end(group_id)
            return self._clients[group_id]

        from langfuse import Langfuse

        client = Langfuse(
            host=self._host,
            public_key=group_id,
            secret_key=group_id,
        )
        self._clients[group_id] = client

        # Evict least-recently-used client if cache is full
        if len(self._clients) > MAX_CLIENT_CACHE_SIZE:
            _, evicted = self._clients.popitem(last=False)
            evicted.flush()
            evicted.shutdown()

        return client

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
        session_id: str | None = None,
        trace_id: str | None = None,
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

        trace_name = model or "llm-proxy"
        valid_trace_id = Langfuse.create_trace_id(seed=trace_id) if trace_id else None
        trace_context = {"trace_id": valid_trace_id} if valid_trace_id else None

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(milliseconds=duration_ms)

        with propagate_attributes(session_id=session_id or None):
            with client.start_as_current_observation(
                name=trace_name,
                as_type="generation",
                model=model,
                input=input_body,
                output=output_body,
                metadata=metadata,
                usage_details=usage_details or None,
                cost_details=cost_details,
                trace_context=trace_context,
                start_time=start_time,
                end_time=end_time,
            ):
                pass

    def trace_search_request(
        self,
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
        client = self._get_client(group_id)

        metadata = {
            "provider": provider,
            "group_id": group_id,
            "status_code": status_code,
            "duration_ms": duration_ms,
            "num_results_requested": num_results,
            "num_results_returned": result_count,
        }
        cost_details = None
        if cost is not None:
            cost_details = {"total": cost}
            metadata["cost"] = cost

        valid_trace_id = Langfuse.create_trace_id(seed=trace_id) if trace_id else None
        trace_context = {"trace_id": valid_trace_id} if valid_trace_id else None

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(milliseconds=duration_ms)

        with propagate_attributes(session_id=session_id or None):
            with client.start_as_current_observation(
                name=f"search:{provider}",
                as_type="span",
                input={"query": query} if query is not None else None,
                output={"result_count": result_count},
                metadata=metadata,
                usage_details={"total": 1, "unit": "SEARCHES"},
                cost_details=cost_details,
                trace_context=trace_context,
                start_time=start_time,
                end_time=end_time,
            ):
                pass

    async def score(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: str | None,
        group_id: str,
    ) -> None:
        """Submit a score via the Langfuse REST API (POST /api/public/scores)."""
        import httpx

        pk = group_id if group_id and group_id != "unknown" else self._public_key
        sk = group_id if group_id and group_id != "unknown" else self._secret_key

        payload: dict = {"traceId": trace_id, "name": name, "value": value}
        if comment:
            payload["comment"] = comment

        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{self._host.rstrip('/')}/api/public/scores",
                json=payload,
                auth=(pk, sk),
            )
            resp.raise_for_status()

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
