import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from starlette.testclient import TestClient


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_settings(**overrides):
    s = MagicMock()
    s.app_name = "TestApp"
    s.debug = False
    s.root_path = ""
    s.allowed_origins = ["*"]
    s.proxy_enabled = False
    s.prompt_hub_enabled = False
    s.chain_hub_enabled = False
    s.llm_hub_enabled = False
    s.dashboard_enabled = False
    s.tracing_enabled = True
    s.billing_enabled = False
    s.version = "0.0.1-test"
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def _app(settings_overrides=None):
    settings_overrides = settings_overrides or {}
    mock_settings = _make_settings(**settings_overrides)
    with patch("app.core.app.settings", mock_settings):
        from app.core.app import create_app
        return create_app(), mock_settings


# ── POST /v1/feedback ─────────────────────────────────────────────────────────

class TestFeedbackEndpoint:
    def _client(self, settings_overrides=None, extra_patches=None):
        """Build a TestClient with all patches active. Returns (client, patches_dict)."""
        mock_settings = _make_settings(**(settings_overrides or {}))
        patches = {"app.core.app.settings": mock_settings}
        patches.update(extra_patches or {})
        return mock_settings, patches

    def test_returns_503_when_tracing_disabled(self):
        mock_settings = _make_settings(tracing_enabled=False)
        with patch("app.core.app.settings", mock_settings):
            from app.core.app import create_app
            client = TestClient(create_app(), raise_server_exceptions=False)
            resp = client.post("/v1/feedback", json={"request_id": "req-123", "score": 1.0})
        assert resp.status_code == 503

    def test_happy_path_returns_ok_and_trace_id(self):
        fake_trace_id = "aaaabbbbccccdddd1111222233334444"
        mock_settings = _make_settings()
        mock_emitter = AsyncMock()

        with patch("app.core.app.settings", mock_settings), \
             patch("app.services.tracing.get_emitter", return_value=mock_emitter), \
             patch("langfuse.Langfuse") as mock_lf_cls:
            mock_lf_cls.create_trace_id.return_value = fake_trace_id
            from app.core.app import create_app
            client = TestClient(create_app(), raise_server_exceptions=False)

            resp = client.post(
                "/v1/feedback",
                json={"request_id": "req-abc", "score": 0.9, "comment": "great"},
                headers={"x-group-id": "org1/user1"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["trace_id"] == fake_trace_id

    def test_score_delegated_with_correct_args(self):
        fake_trace_id = "aaaabbbbccccdddd1111222233334444"
        mock_settings = _make_settings()
        mock_emitter = AsyncMock()

        with patch("app.core.app.settings", mock_settings), \
             patch("app.services.tracing.get_emitter", return_value=mock_emitter), \
             patch("langfuse.Langfuse") as mock_lf_cls:
            mock_lf_cls.create_trace_id.return_value = fake_trace_id
            from app.core.app import create_app
            client = TestClient(create_app(), raise_server_exceptions=False)

            client.post(
                "/v1/feedback",
                json={"request_id": "req-abc", "score": -1.0, "name": "thumbs", "comment": "wrong answer"},
                headers={"x-group-id": "org1/user1"},
            )

        mock_emitter.score.assert_awaited_once_with(
            trace_id=fake_trace_id,
            name="thumbs",
            value=-1.0,
            comment="wrong answer",
            group_id="org1/user1",
        )

    def test_default_group_id_when_header_absent(self):
        mock_settings = _make_settings()
        mock_emitter = AsyncMock()

        with patch("app.core.app.settings", mock_settings), \
             patch("app.services.tracing.get_emitter", return_value=mock_emitter), \
             patch("langfuse.Langfuse") as mock_lf_cls:
            mock_lf_cls.create_trace_id.return_value = "aaaa1111"
            from app.core.app import create_app
            client = TestClient(create_app(), raise_server_exceptions=False)

            resp = client.post("/v1/feedback", json={"request_id": "req-xyz", "score": 1.0})

        assert resp.status_code == 200
        _, kwargs = mock_emitter.score.call_args
        assert kwargs["group_id"] == "unknown"

    def test_emitter_error_does_not_crash_endpoint(self):
        mock_settings = _make_settings()
        mock_emitter = AsyncMock()
        mock_emitter.score.side_effect = RuntimeError("langfuse down")

        with patch("app.core.app.settings", mock_settings), \
             patch("app.services.tracing.get_emitter", return_value=mock_emitter), \
             patch("langfuse.Langfuse") as mock_lf_cls, \
             patch("app.services.tracing.logger"):
            mock_lf_cls.create_trace_id.return_value = "tid"
            from app.core.app import create_app
            client = TestClient(create_app(), raise_server_exceptions=False)

            resp = client.post("/v1/feedback", json={"request_id": "req-err", "score": 1.0})

        assert resp.status_code == 200


# ── score_trace facade ────────────────────────────────────────────────────────

class TestScoreTraceFacade:
    @pytest.fixture(autouse=True)
    def reset_emitter(self):
        import app.services.tracing as mod
        mod._emitter = None
        yield
        mod._emitter = None

    @pytest.mark.asyncio
    async def test_returns_empty_string_when_disabled(self):
        from app.services.tracing import score_trace

        with patch("app.services.tracing.get_emitter", return_value=None):
            result = await score_trace("req-1", "user_feedback", 1.0, None, "group-1")
        assert result == ""

    @pytest.mark.asyncio
    async def test_derives_trace_id_from_request_id(self):
        from app.services.tracing import score_trace

        mock_emitter = AsyncMock()
        with patch("app.services.tracing.get_emitter", return_value=mock_emitter), \
             patch("langfuse.Langfuse") as mock_lf_cls:
            mock_lf_cls.create_trace_id.return_value = "derived-id"
            result = await score_trace("req-seed", "user_feedback", 1.0, None, "g1")

        mock_lf_cls.create_trace_id.assert_called_once_with(seed="req-seed")
        assert result == "derived-id"

    @pytest.mark.asyncio
    async def test_passes_all_args_to_emitter(self):
        from app.services.tracing import score_trace

        mock_emitter = AsyncMock()
        with patch("app.services.tracing.get_emitter", return_value=mock_emitter), \
             patch("langfuse.Langfuse") as mock_lf_cls:
            mock_lf_cls.create_trace_id.return_value = "tid-42"
            await score_trace("req-42", "stars", 4.0, "very helpful", "org/user")

        mock_emitter.score.assert_awaited_once_with(
            trace_id="tid-42",
            name="stars",
            value=4.0,
            comment="very helpful",
            group_id="org/user",
        )


# ── LangfuseEmitter.score ─────────────────────────────────────────────────────

class TestLangfuseEmitterScore:
    @pytest.mark.asyncio
    async def test_posts_to_langfuse_scores_endpoint(self):
        from app.services.langfuse_tracing import LangfuseEmitter
        import httpx

        mock_langfuse = MagicMock()
        with patch.dict("sys.modules", {"langfuse": MagicMock(Langfuse=MagicMock(return_value=mock_langfuse))}):
            emitter = LangfuseEmitter()
            emitter._host = "http://llogr:5000"
            emitter._public_key = "lf-pk"
            emitter._secret_key = "lf-sk"

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_async_client = AsyncMock()
        mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_async_client.__aexit__ = AsyncMock(return_value=False)
        mock_async_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_async_client):
            await emitter.score("trace-id-1", "user_feedback", 0.8, "good", "group-abc")

        mock_async_client.post.assert_awaited_once_with(
            "http://llogr:5000/api/public/scores",
            json={"traceId": "trace-id-1", "name": "user_feedback", "value": 0.8, "comment": "good"},
            auth=("group-abc", "group-abc"),
        )

    @pytest.mark.asyncio
    async def test_uses_default_keys_for_unknown_group(self):
        from app.services.langfuse_tracing import LangfuseEmitter

        mock_langfuse = MagicMock()
        with patch.dict("sys.modules", {"langfuse": MagicMock(Langfuse=MagicMock(return_value=mock_langfuse))}):
            emitter = LangfuseEmitter()
            emitter._host = "http://llogr:5000"
            emitter._public_key = "lf-pk-ai-suite"
            emitter._secret_key = "lf-sk-ai-suite"

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_async_client = AsyncMock()
        mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_async_client.__aexit__ = AsyncMock(return_value=False)
        mock_async_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_async_client):
            await emitter.score("tid", "feedback", 1.0, None, "unknown")

        _, kwargs = mock_async_client.post.call_args
        assert kwargs["auth"] == ("lf-pk-ai-suite", "lf-sk-ai-suite")

    @pytest.mark.asyncio
    async def test_comment_omitted_when_none(self):
        from app.services.langfuse_tracing import LangfuseEmitter

        mock_langfuse = MagicMock()
        with patch.dict("sys.modules", {"langfuse": MagicMock(Langfuse=MagicMock(return_value=mock_langfuse))}):
            emitter = LangfuseEmitter()
            emitter._host = "http://host"
            emitter._public_key = "pk"
            emitter._secret_key = "sk"

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_async_client = AsyncMock()
        mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_async_client.__aexit__ = AsyncMock(return_value=False)
        mock_async_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_async_client):
            await emitter.score("tid", "feedback", 1.0, None, "g1")

        _, kwargs = mock_async_client.post.call_args
        assert "comment" not in kwargs["json"]
