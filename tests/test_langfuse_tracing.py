import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def reset_client():
    """Reset the module-level _client before each test."""
    import app.services.langfuse_tracing as mod
    mod._client = None
    yield
    mod._client = None


class TestGetClient:
    def test_returns_none_when_disabled(self):
        from app.services.langfuse_tracing import get_client

        with patch("app.services.langfuse_tracing.settings") as mock_settings:
            mock_settings.langfuse_enabled = False
            assert get_client() is None

    def test_creates_client_when_enabled(self):
        from app.services.langfuse_tracing import get_client
        import app.services.langfuse_tracing as mod

        mock_langfuse_instance = MagicMock()
        with patch("app.services.langfuse_tracing.settings") as mock_settings, \
             patch.dict("sys.modules", {"langfuse": MagicMock(Langfuse=MagicMock(return_value=mock_langfuse_instance))}):
            mock_settings.langfuse_enabled = True
            client = get_client()
            assert client is mock_langfuse_instance

    def test_returns_same_client_on_second_call(self):
        from app.services.langfuse_tracing import get_client
        import app.services.langfuse_tracing as mod

        mock_langfuse_instance = MagicMock()
        with patch("app.services.langfuse_tracing.settings") as mock_settings, \
             patch.dict("sys.modules", {"langfuse": MagicMock(Langfuse=MagicMock(return_value=mock_langfuse_instance))}):
            mock_settings.langfuse_enabled = True
            client1 = get_client()
            client2 = get_client()
            assert client1 is client2


class TestShutdown:
    def test_shutdown_flushes_and_clears(self):
        from app.services.langfuse_tracing import shutdown
        import app.services.langfuse_tracing as mod

        mock_client = MagicMock()
        mod._client = mock_client

        shutdown()

        mock_client.flush.assert_called_once()
        mock_client.shutdown.assert_called_once()
        assert mod._client is None

    def test_shutdown_noop_when_no_client(self):
        from app.services.langfuse_tracing import shutdown
        import app.services.langfuse_tracing as mod

        mod._client = None
        shutdown()  # Should not raise


class TestTraceProxyRequest:
    def test_noop_when_disabled(self):
        from app.services.langfuse_tracing import trace_proxy_request

        with patch("app.services.langfuse_tracing.get_client", return_value=None):
            # Should not raise
            trace_proxy_request(
                model="test-model",
                provider=None,
                input_body={"messages": [{"role": "user", "content": "hi"}]},
                output_body={"choices": []},
                status_code=200,
                usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
                duration_ms=100.0,
                group_id="group-1",
                is_streaming=False,
            )

    def test_creates_trace_and_generation(self):
        from app.services.langfuse_tracing import trace_proxy_request

        mock_client = MagicMock()
        mock_trace = MagicMock()
        mock_client.trace.return_value = mock_trace

        input_body = {"messages": [{"role": "user", "content": "hello"}]}
        output_body = {"choices": [{"message": {"content": "world"}}]}
        usage = {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}

        with patch("app.services.langfuse_tracing.get_client", return_value=mock_client), \
             patch("app.services.langfuse_tracing.settings") as mock_settings:
            mock_settings.langfuse_log_io = True

            trace_proxy_request(
                model="test-model",
                provider="gigachat",
                input_body=input_body,
                output_body=output_body,
                status_code=200,
                usage=usage,
                duration_ms=150.5,
                group_id="group-1",
                is_streaming=False,
            )

        mock_client.trace.assert_called_once()
        trace_kwargs = mock_client.trace.call_args[1]
        assert trace_kwargs["name"] == "llm-proxy"
        assert trace_kwargs["metadata"]["provider"] == "gigachat"
        assert trace_kwargs["metadata"]["group_id"] == "group-1"
        assert trace_kwargs["metadata"]["is_streaming"] is False

        mock_trace.generation.assert_called_once()
        gen_kwargs = mock_trace.generation.call_args[1]
        assert gen_kwargs["model"] == "test-model"
        assert gen_kwargs["input"] == input_body
        assert gen_kwargs["output"] == output_body
        assert gen_kwargs["usage"] == usage

    def test_strips_io_when_log_io_false(self):
        from app.services.langfuse_tracing import trace_proxy_request

        mock_client = MagicMock()
        mock_trace = MagicMock()
        mock_client.trace.return_value = mock_trace

        with patch("app.services.langfuse_tracing.get_client", return_value=mock_client), \
             patch("app.services.langfuse_tracing.settings") as mock_settings:
            mock_settings.langfuse_log_io = False

            trace_proxy_request(
                model="test-model",
                provider=None,
                input_body={"messages": [{"role": "user", "content": "secret"}]},
                output_body={"choices": [{"message": {"content": "also secret"}}]},
                status_code=200,
                usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
                duration_ms=100.0,
                group_id="group-1",
                is_streaming=False,
            )

        gen_kwargs = mock_trace.generation.call_args[1]
        assert gen_kwargs["input"] is None
        assert gen_kwargs["output"] is None
        # But model and usage should still be present
        assert gen_kwargs["model"] == "test-model"
        assert gen_kwargs["usage"]["total_tokens"] == 8

    def test_handles_exception_gracefully(self):
        from app.services.langfuse_tracing import trace_proxy_request

        mock_client = MagicMock()
        mock_client.trace.side_effect = RuntimeError("connection failed")

        with patch("app.services.langfuse_tracing.get_client", return_value=mock_client), \
             patch("app.services.langfuse_tracing.settings") as mock_settings:
            mock_settings.langfuse_log_io = True

            # Should not raise
            trace_proxy_request(
                model="test-model",
                provider=None,
                input_body=None,
                output_body=None,
                status_code=200,
                usage=None,
                duration_ms=100.0,
                group_id="group-1",
                is_streaming=False,
            )
