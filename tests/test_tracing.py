import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def reset_emitter():
    """Reset the module-level _emitter before each test."""
    import app.services.tracing as mod
    mod._emitter = None
    yield
    mod._emitter = None


class TestGetEmitter:
    def test_returns_none_when_disabled(self):
        from app.services.tracing import get_emitter

        with patch("app.services.tracing.settings") as mock_settings:
            mock_settings.tracing_enabled = False
            assert get_emitter() is None

    def test_returns_langfuse_emitter_when_enabled(self):
        from app.services.tracing import get_emitter

        mock_emitter = MagicMock()
        with patch("app.services.tracing.settings") as mock_settings, \
             patch("app.services.tracing.logger"), \
             patch.dict("sys.modules", {"langfuse": MagicMock()}), \
             patch("app.services.langfuse_tracing.LangfuseEmitter", return_value=mock_emitter) as mock_cls:
            mock_settings.tracing_enabled = True
            mock_settings.tracing_backend = "langfuse"

            emitter = get_emitter()
            assert emitter is mock_emitter

    def test_returns_same_emitter_on_second_call(self):
        from app.services.tracing import get_emitter
        import app.services.tracing as mod

        sentinel = MagicMock()
        mod._emitter = sentinel

        with patch("app.services.tracing.settings") as mock_settings:
            mock_settings.tracing_enabled = True
            assert get_emitter() is sentinel
            assert get_emitter() is sentinel

    def test_returns_none_for_unknown_backend(self):
        from app.services.tracing import get_emitter

        with patch("app.services.tracing.settings") as mock_settings, \
             patch("app.services.tracing.logger"):
            mock_settings.tracing_enabled = True
            mock_settings.tracing_backend = "unknown"
            assert get_emitter() is None


class TestTraceProxyRequest:
    def test_noop_when_disabled(self):
        from app.services.tracing import trace_proxy_request

        with patch("app.services.tracing.get_emitter", return_value=None):
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

    def test_delegates_to_emitter(self):
        from app.services.tracing import trace_proxy_request

        mock_emitter = MagicMock()
        input_body = {"messages": [{"role": "user", "content": "hello"}]}
        output_body = {"choices": [{"message": {"content": "world"}}]}
        usage = {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}

        with patch("app.services.tracing.get_emitter", return_value=mock_emitter), \
             patch("app.services.tracing.settings") as mock_settings:
            mock_settings.tracing_log_io = True

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

        mock_emitter.trace_proxy_request.assert_called_once_with(
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

    def test_strips_io_when_log_io_false(self):
        from app.services.tracing import trace_proxy_request

        mock_emitter = MagicMock()

        with patch("app.services.tracing.get_emitter", return_value=mock_emitter), \
             patch("app.services.tracing.settings") as mock_settings:
            mock_settings.tracing_log_io = False

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

        call_kwargs = mock_emitter.trace_proxy_request.call_args[1]
        assert call_kwargs["input_body"] is None
        assert call_kwargs["output_body"] is None
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["usage"]["total_tokens"] == 8

    def test_handles_emitter_exception_gracefully(self):
        from app.services.tracing import trace_proxy_request

        mock_emitter = MagicMock()
        mock_emitter.trace_proxy_request.side_effect = RuntimeError("connection failed")

        with patch("app.services.tracing.get_emitter", return_value=mock_emitter), \
             patch("app.services.tracing.settings") as mock_settings, \
             patch("app.services.tracing.logger"):
            mock_settings.tracing_log_io = True

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


class TestGetLangchainCallback:
    def test_returns_handler(self):
        from app.services.langfuse_tracing import LangfuseEmitter

        mock_handler = MagicMock()
        mock_langfuse = MagicMock()

        with patch.dict("sys.modules", {
            "langfuse": MagicMock(Langfuse=MagicMock(return_value=mock_langfuse)),
            "langfuse.callback": MagicMock(CallbackHandler=MagicMock(return_value=mock_handler)),
        }):
            emitter = LangfuseEmitter()
            cb = emitter.get_langchain_callback(
                trace_name="chain-execution",
                metadata={"chain_name": "test", "group_id": "g1"},
            )
            assert cb is mock_handler


class TestShutdown:
    def test_delegates_to_emitter(self):
        from app.services.tracing import shutdown
        import app.services.tracing as mod

        mock_emitter = MagicMock()
        mod._emitter = mock_emitter

        shutdown()

        mock_emitter.shutdown.assert_called_once()
        assert mod._emitter is None

    def test_noop_when_no_emitter(self):
        from app.services.tracing import shutdown
        import app.services.tracing as mod

        mod._emitter = None
        shutdown()  # Should not raise

    def test_handles_shutdown_exception(self):
        from app.services.tracing import shutdown
        import app.services.tracing as mod

        mock_emitter = MagicMock()
        mock_emitter.shutdown.side_effect = RuntimeError("fail")
        mod._emitter = mock_emitter

        with patch("app.services.tracing.logger"):
            shutdown()  # Should not raise

        assert mod._emitter is None
