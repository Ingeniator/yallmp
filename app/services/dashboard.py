from datetime import datetime, timezone
from prometheus_client import generate_latest
from prometheus_client.parser import text_string_to_metric_families


def parse_metrics_to_dict(registry):
    """Parse Prometheus registry into classified metric entries."""
    raw = generate_latest(registry).decode("utf-8")
    token_usage = []
    http_requests = []
    http_duration = []

    for family in text_string_to_metric_families(raw):
        for sample in family.samples:
            name = sample.name
            labels = dict(sample.labels)
            value = sample.value

            if name.startswith("llm_") and name.endswith("_token_usage_total"):
                kind = name.replace("llm_", "").replace("_token_usage_total", "")
                token_usage.append({"metric": kind, "value": value, **labels})
            elif name == "http_requests_total":
                http_requests.append({"value": value, **labels})
            elif name.startswith("http_request_duration_seconds"):
                suffix = name.replace("http_request_duration_seconds_", "")
                http_duration.append({"stat": suffix, "value": value, **labels})

    return {
        "token_usage": token_usage,
        "http_requests": http_requests,
        "http_duration": http_duration,
    }


def _aggregate_tokens_by(label_key, entries):
    """Group total-token entries by a label key and sum values."""
    agg = {}
    for e in entries:
        if e.get("metric") != "total":
            continue
        key = e.get(label_key, "unknown")
        agg[key] = agg.get(key, 0) + e["value"]
    return agg


def _aggregate_requests_by_endpoint(entries):
    """Sum request counts by endpoint."""
    agg = {}
    for e in entries:
        ep = e.get("endpoint", "unknown")
        agg[ep] = agg.get(ep, 0) + e["value"]
    return agg


def _compute_avg_duration(entries):
    """Compute average duration per endpoint from histogram sum/count."""
    sums = {}
    counts = {}
    for e in entries:
        ep = e.get("endpoint", "unknown")
        if e.get("stat") == "sum":
            sums[ep] = sums.get(ep, 0) + e["value"]
        elif e.get("stat") == "count":
            counts[ep] = counts.get(ep, 0) + e["value"]

    avg = {}
    for ep in sums:
        c = counts.get(ep, 0)
        avg[ep] = round(sums[ep] / c, 6) if c > 0 else 0.0
    return avg


def get_dashboard_json(registry):
    """Build complete dashboard payload with raw data and pre-aggregated summary."""
    data = parse_metrics_to_dict(registry)
    summary = {
        "tokens_by_model": _aggregate_tokens_by("model", data["token_usage"]),
        "tokens_by_group": _aggregate_tokens_by("group_id", data["token_usage"]),
        "tokens_by_type": _aggregate_tokens_by("type", data["token_usage"]),
        "requests_by_endpoint": _aggregate_requests_by_endpoint(data["http_requests"]),
        "avg_duration_by_endpoint": _compute_avg_duration(data["http_duration"]),
    }
    return {
        **data,
        "summary": summary,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
