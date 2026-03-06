import pytest
from unittest.mock import patch, MagicMock

from app.services.dashboard import (
    parse_metrics_to_dict,
    _aggregate_tokens_by,
    _aggregate_cost_by,
    _compute_avg_duration,
    get_dashboard_json,
)


# -- Fake registry that returns canned Prometheus text --

SAMPLE_METRICS = """\
# HELP llm_total_token_usage_total Total LLM tokens used by the prompt
# TYPE llm_total_token_usage_total counter
llm_total_token_usage_total{type="chain",name="summarize",group_id="g1",model="gpt-4"} 150.0
llm_total_token_usage_total{type="prompt",name="greet",group_id="g2",model="gpt-3.5"} 80.0
# HELP llm_prompt_token_usage_total Prompt LLM tokens used by the prompt
# TYPE llm_prompt_token_usage_total counter
llm_prompt_token_usage_total{type="chain",name="summarize",group_id="g1",model="gpt-4"} 100.0
# HELP llm_completion_token_usage_total Completion LLM tokens used by the prompt
# TYPE llm_completion_token_usage_total counter
llm_completion_token_usage_total{type="chain",name="summarize",group_id="g1",model="gpt-4"} 50.0
# HELP http_requests_total Total number of HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="GET",endpoint="/health",status_code="200"} 10.0
http_requests_total{method="POST",endpoint="/llm/v1/chat/completions",status_code="200"} 5.0
# HELP http_request_duration_seconds Histogram of request processing time
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_sum{method="GET",endpoint="/health"} 0.5
http_request_duration_seconds_count{method="GET",endpoint="/health"} 10.0
http_request_duration_seconds_sum{method="POST",endpoint="/llm/v1/chat/completions"} 2.5
http_request_duration_seconds_count{method="POST",endpoint="/llm/v1/chat/completions"} 5.0
# HELP llm_cost_total_total Total estimated cost in provider currency
# TYPE llm_cost_total_total counter
llm_cost_total_total{provider="openai",currency="USD",model="gpt-4",group_id="g1"} 0.0045
llm_cost_total_total{provider="openai",currency="USD",model="gpt-3.5",group_id="g2"} 0.0002
llm_cost_total_total{provider="anthropic",currency="USD",model="claude-3",group_id="g1"} 0.003
"""


def _fake_registry():
    """Return a mock registry whose generate_latest returns SAMPLE_METRICS."""
    reg = MagicMock()
    return reg


@patch("app.services.dashboard.generate_latest", return_value=SAMPLE_METRICS.encode("utf-8"))
def test_parse_metrics_to_dict(mock_gen):
    data = parse_metrics_to_dict(_fake_registry())

    assert len(data["token_usage"]) == 4  # 2 total + 1 prompt + 1 completion
    assert len(data["http_requests"]) == 2
    assert len(data["http_duration"]) == 4  # 2 sum + 2 count
    assert len(data["cost"]) == 3

    # Check classification
    metrics = {e["metric"] for e in data["token_usage"]}
    assert "total" in metrics
    assert "prompt" in metrics
    assert "completion" in metrics

    # Check cost entries have expected labels
    providers = {e["provider"] for e in data["cost"]}
    assert "openai" in providers
    assert "anthropic" in providers


@patch("app.services.dashboard.generate_latest", return_value=SAMPLE_METRICS.encode("utf-8"))
def test_aggregate_tokens_by(mock_gen):
    data = parse_metrics_to_dict(_fake_registry())

    by_model = _aggregate_tokens_by("model", data["token_usage"])
    assert by_model["gpt-4"] == 150.0
    assert by_model["gpt-3.5"] == 80.0

    by_group = _aggregate_tokens_by("group_id", data["token_usage"])
    assert by_group["g1"] == 150.0
    assert by_group["g2"] == 80.0


@patch("app.services.dashboard.generate_latest", return_value=SAMPLE_METRICS.encode("utf-8"))
def test_aggregate_cost_by(mock_gen):
    data = parse_metrics_to_dict(_fake_registry())

    by_provider = _aggregate_cost_by("provider", data["cost"])
    assert by_provider["openai"] == pytest.approx(0.0047, abs=1e-6)
    assert by_provider["anthropic"] == pytest.approx(0.003, abs=1e-6)

    by_model = _aggregate_cost_by("model", data["cost"])
    assert by_model["gpt-4"] == pytest.approx(0.0045, abs=1e-6)
    assert by_model["claude-3"] == pytest.approx(0.003, abs=1e-6)


@patch("app.services.dashboard.generate_latest", return_value=SAMPLE_METRICS.encode("utf-8"))
def test_compute_avg_duration(mock_gen):
    data = parse_metrics_to_dict(_fake_registry())
    avg = _compute_avg_duration(data["http_duration"])

    assert avg["/health"] == pytest.approx(0.05, abs=1e-4)
    assert avg["/llm/v1/chat/completions"] == pytest.approx(0.5, abs=1e-4)


def test_compute_avg_duration_division_by_zero():
    """When count is 0, average should be 0.0 (no division error)."""
    entries = [
        {"stat": "sum", "endpoint": "/test", "value": 1.5},
        {"stat": "count", "endpoint": "/test", "value": 0.0},
    ]
    avg = _compute_avg_duration(entries)
    assert avg["/test"] == 0.0


def _mock_settings():
    class S:
        app_name = "TestApp"
        debug = False
        root_path = ""
        allowed_origins = ["*"]
        proxy_enabled = False
        prompt_hub_enabled = False
        chain_hub_enabled = False
        llm_hub_enabled = False
        dashboard_enabled = True
        dashboard_metrics_backend = "local"
        dashboard_prometheus_url = ""
        dashboard_prometheus_timeout = 10
        version = "0.0.1-test"
    return S()


def test_dashboard_html_endpoint():
    with patch("app.core.app.settings", _mock_settings()):
        from app.core.app import create_app
        from starlette.testclient import TestClient

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/dashboard")

    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Metrics Dashboard" in resp.text


def test_dashboard_api_metrics_endpoint():
    with patch("app.core.app.settings", _mock_settings()), \
         patch("app.services.dashboard.settings", _mock_settings()), \
         patch("app.services.dashboard._load_endpoint_patterns", return_value=[]):
        from app.core.app import create_app
        from starlette.testclient import TestClient

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/dashboard/api/metrics")

    assert resp.status_code == 200
    data = resp.json()
    assert "token_usage" in data
    assert "http_requests" in data
    assert "http_duration" in data
    assert "cost" in data
    assert "summary" in data
    assert "timestamp" in data


@pytest.mark.asyncio
@patch("app.services.dashboard.settings", _mock_settings())
@patch("app.services.dashboard._load_endpoint_patterns", return_value=[])
@patch("app.services.dashboard.generate_latest", return_value=SAMPLE_METRICS.encode("utf-8"))
async def test_get_dashboard_json_summary(mock_gen, mock_patterns):
    result = await get_dashboard_json(registry=_fake_registry())

    assert "summary" in result
    s = result["summary"]
    assert s["tokens_by_model"]["gpt-4"] == 150.0
    assert s["tokens_by_group"]["g2"] == 80.0
    assert s["tokens_by_type"]["chain"] == 150.0
    assert s["requests_by_endpoint"]["/health"] == 10.0
    assert s["avg_duration_by_endpoint"]["/health"] == pytest.approx(0.05, abs=1e-4)
    assert s["cost_by_model"]["gpt-4"] == pytest.approx(0.0045, abs=1e-6)
    assert s["cost_by_provider"]["openai"] == pytest.approx(0.0047, abs=1e-6)
    assert s["cost_by_group"]["g1"] == pytest.approx(0.0075, abs=1e-6)
    assert "timestamp" in result
