import warnings

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import os


class AppSettings(BaseSettings):
    app_name: str = "LLM-Proxy"
    root_path: str = "/ai"
    metrics_path: str = "/tmp/metrics"
    debug: bool = False
    workers: int = 1
    timeout_keep_alive: int = 600
    max_connections: int = 100  # Maximum simultaneous connections
    max_keepalive_connections: int = 20  # Keepalive connections
    host: str = "0.0.0.0"
    port: int = 5000
    allowed_origins: list[str] = ["*"]

    # Fake LLM
    fake_llm_host: str = "0.0.0.0"
    fake_llm_port: int = 5001

    # Feature toggles
    proxy_enabled: bool = False  # /llm routes, served by the LLM hub providers
    prompt_hub_enabled: bool = False
    chain_hub_enabled: bool = False
    llm_hub_enabled: bool = False  # deprecated alias of proxy_enabled
    dashboard_enabled: bool = False
    dashboard_metrics_backend: str = "local"  # "local" or "prometheus"
    dashboard_prometheus_url: str = ""  # e.g. "http://prometheus:9090"
    dashboard_prometheus_timeout: int = 10
    dashboard_prometheus_user: str = ""
    dashboard_prometheus_password: str = ""
    dashboard_prometheus_verify_ssl: bool = False
    dashboard_prometheus_ca_bundle: str = ""

    # LLM Hub
    llm_hub_directory: str = "data/llm_hub"

    # Tracing
    tracing_enabled: bool = False
    tracing_log_io: bool = (
        True  # True = log messages+response content; False = metadata only
    )
    tracing_backend: str = "langfuse"  # selects which TraceEmitter to instantiate
    tracing_host: str | None = None  # e.g. http://llogr:8000
    tracing_public_key: str = "default"
    tracing_secret_key: str = "default"
    tracing_environment: str | None = None  # e.g. "production", "staging"
    tracing_release: str | None = None  # e.g. git commit SHA or semver tag

    # OpenTelemetry distributed tracing (separate from the business-level
    # Langfuse tracing above) — exports spans for HTTP requests and outbound
    # httpx calls to an OTLP collector, e.g. Tempo.
    otel_enabled: bool = False
    otel_service_name: str = "yallmp"
    otel_exporter_endpoint: str = "http://tempo:4318"  # OTLP/HTTP receiver base URL

    # Logging
    log_level: str = "INFO"
    log_file: str | None = None
    silence_probes: bool = True
    debug_network_check: bool = True

    # Proxy
    proxy_exclude_headers: str = (
        "host,authorization,cookie,x-forwarded-*,jwt-*,content-length"
    )

    @model_validator(mode="after")
    def _merge_llm_hub_flag(self) -> "AppSettings":
        # The hub is the only /llm backend now, so the two flags mean the same thing.
        if self.llm_hub_enabled:
            warnings.warn(
                "llm_hub_enabled is deprecated: use proxy_enabled",
                DeprecationWarning,
                stacklevel=2,
            )
            self.proxy_enabled = True
        return self

    proxy_max_retries: int = 5  # Number of retries before circuit breaker
    proxy_base_delay: float = 0.5  # Base delay in seconds
    proxy_backoff_factor: float = 2.0  # Exponential backoff multiplier
    # Circuit breaker configuration
    proxy_failure_threshold: int = (
        0  # 0 if circuit breaker disabled. Number of failures before tripping
    )
    proxy_recovery_time: int = 30  # Cooldown period (seconds)
    proxy_window_size: int = 60  # Sliding window size in seconds

    # Prompt hub settings
    prompt_hub_directory: str = "data/prompt_hub"

    # Chain hub settings
    chain_hub_directory: str = "data/chain_hub"
    chain_default_base_url: str = ""
    chain_default_model_name: str = ""
    chain_default_ca_bundle_file: str = ""
    chain_default_verify_ssl_certs: bool = False
    chain_default_cert_file: str = ""
    chain_default_key_file: str = ""
    chain_default_auth_url: str = ""
    chain_default_credentials: str = ""
    chain_default_scope: str = "GIGACHAT_API_CORP"
    chain_default_available_chat_models: list[str] = [
        "Gigachat:latest",
        "GigaChat-Pro",
        "GigaChat-Max",
        "Gigachat-2:latest",
        "DeepSeek-R1",
    ]
    chain_default_json_file: str = "./data/langchain/default.json"

    # Dev role switcher (docker-compose only — never enable in production)
    dashboard_dev_role_switcher: bool = False

    # Search Hub
    search_hub_enabled: bool = False
    search_hub_directory: str = "data/search_hub"

    # Billing
    billing_enabled: bool = False
    billing_redis_url: str = "redis://tagme-x-redis:6379/1"
    billing_limits_path: str = "data/billing/limits.yaml"
    billing_sync_url: str = (
        ""  # llogr base URL for Redis cache seeding; falls back to tracing_host
    )
    billing_sync_interval: int = 300  # seconds between periodic syncs

    version: str = "0.1.0"

    model_config = SettingsConfigDict(
        env_prefix="LLM_", env_file=".env", env_file_encoding="utf-8", extra="allow"
    )


# Settings of the removed single-provider proxy. They are ignored now; the
# upstream must be configured as a provider JSON in llm_hub_directory.
REMOVED_PROXY_SETTINGS = (
    "proxy_target_url",
    "proxy_authorization_type",
    "proxy_api_key",
    "proxy_api_cert_path",
    "proxy_api_cert_key_path",
    "proxy_oidc_authorization_url",
    "proxy_oidc_credentials",
    "proxy_verify_ssl",
    "proxy_ca_bundle_path",
    "proxy_connect_timeout",
    "proxy_read_timeout",
    "proxy_write_timeout",
    "proxy_pool_timeout",
    "proxy_pricing_config",
    "proxy_pricing_endpoint",
    "proxy_pricing_prefix",
    "proxy_pricing_currency",
)


def removed_proxy_settings_in_use(s: AppSettings) -> list[str]:
    """Names of removed proxy settings still present in the environment or .env."""
    # Unknown keys from .env land in model_extra with their LLM_ prefix kept.
    present = {k.lower() for k in (getattr(s, "model_extra", None) or {})} | {
        k.lower() for k in os.environ
    }
    return [name for name in REMOVED_PROXY_SETTINGS if f"llm_{name}" in present]


# Load settings
settings = AppSettings()

os.environ["LLM_CHAIN_DEFAULT_BASE_URL"] = settings.chain_default_base_url
os.environ["LLM_CHAIN_DEFAULT_MODEL_NAME"] = settings.chain_default_model_name
os.environ["PROMETHEUS_MULTIPROC_DIR"] = settings.metrics_path
