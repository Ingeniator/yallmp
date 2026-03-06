import json
import re
from datetime import datetime, timezone
from pathlib import Path
from prometheus_client import generate_latest
from prometheus_client.parser import text_string_to_metric_families

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "dashboard" / "dashboard.json"


def _load_config():
    """Load dashboard config from JSON file."""
    try:
        return json.loads(_CONFIG_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _load_endpoint_patterns():
    """Load endpoint regex patterns from dashboard config."""
    return _load_config().get("endpoint_patterns", [])


def _load_table_columns():
    """Load visible table columns config.

    Returns a dict like {"token_usage": ["metric", "group_id", ...], ...}.
    If not configured, returns empty dict (frontend shows all columns).
    """
    return _load_config().get("table_columns", {})


def _matches_any_pattern(endpoint, patterns):
    """Return True if endpoint matches any of the regex patterns."""
    return any(re.fullmatch(p, endpoint) for p in patterns)


def _filter_by_endpoint(entries, patterns):
    """Keep only entries whose 'endpoint' label matches a pattern."""
    if not patterns:
        return entries
    return [e for e in entries if _matches_any_pattern(e.get("endpoint", ""), patterns)]


def _filter_by_group(entries, group_id, is_org_admin):
    """Filter entries by group_id label.

    - group_id is None  → no filtering (show all)
    - is_org_admin and group_id has org prefix → prefix match on "org/"
    - otherwise → exact match on group_id
    """
    if not group_id:
        return entries
    if is_org_admin and "/" in group_id:
        org_prefix = group_id.split("/", 1)[0] + "/"
        return [e for e in entries if e.get("group_id", "").startswith(org_prefix)]
    return [e for e in entries if e.get("group_id", "") == group_id]


def parse_metrics_to_dict(registry):
    """Parse Prometheus registry into classified metric entries."""
    raw = generate_latest(registry).decode("utf-8")
    token_usage = []
    http_requests = []
    http_duration = []
    cost = []

    for family in text_string_to_metric_families(raw):
        for sample in family.samples:
            name = sample.name
            labels = dict(sample.labels)
            value = sample.value

            if name.startswith("llm_") and name.endswith("_token_usage_total"):
                kind = name.replace("llm_", "").replace("_token_usage_total", "")
                token_usage.append({"metric": kind, "value": value, **labels})
            elif name == "llm_cost_total":
                cost.append({"value": value, **labels})
            elif name == "http_requests_total":
                http_requests.append({"value": value, **labels})
            elif name.startswith("http_request_duration_seconds"):
                suffix = name.replace("http_request_duration_seconds_", "")
                http_duration.append({"stat": suffix, "value": value, **labels})

    return {
        "token_usage": token_usage,
        "http_requests": http_requests,
        "http_duration": http_duration,
        "cost": cost,
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


def _aggregate_cost_by(label_key, entries):
    """Group cost entries by a label key and sum values."""
    agg = {}
    for e in entries:
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


def get_dashboard_json(registry, group_id=None, is_org_admin=False):
    """Build complete dashboard payload with raw data and pre-aggregated summary.

    Args:
        registry: Prometheus CollectorRegistry
        group_id: Optional group_id from x-group-id header for access scoping.
        is_org_admin: True when x-role header is ORG_ADMIN (shows whole org).
    """
    data = parse_metrics_to_dict(registry)

    patterns = _load_endpoint_patterns()
    filtered_requests = _filter_by_group(_filter_by_endpoint(data["http_requests"], patterns), group_id, is_org_admin)
    filtered_duration = _filter_by_group(_filter_by_endpoint(data["http_duration"], patterns), group_id, is_org_admin)

    filtered_tokens = _filter_by_group(data["token_usage"], group_id, is_org_admin)
    filtered_cost = _filter_by_group(data["cost"], group_id, is_org_admin)

    summary = {
        "tokens_by_model": _aggregate_tokens_by("model", filtered_tokens),
        "tokens_by_group": _aggregate_tokens_by("group_id", filtered_tokens),
        "tokens_by_type": _aggregate_tokens_by("type", filtered_tokens),
        "requests_by_endpoint": _aggregate_requests_by_endpoint(filtered_requests),
        "avg_duration_by_endpoint": _compute_avg_duration(filtered_duration),
        "cost_by_model": _aggregate_cost_by("model", filtered_cost),
        "cost_by_provider": _aggregate_cost_by("provider", filtered_cost),
        "cost_by_group": _aggregate_cost_by("group_id", filtered_cost),
    }
    return {
        "token_usage": filtered_tokens,
        "http_requests": filtered_requests,
        "http_duration": filtered_duration,
        "cost": filtered_cost,
        "summary": summary,
        "table_columns": _load_table_columns(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
