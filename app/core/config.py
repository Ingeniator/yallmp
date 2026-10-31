import warnings
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import os


class AppSettings(BaseSettings):
    app_name: str = "LLM-Proxy"
    root_path: str = "/ai"
    metrics_path: str = "/tmp/metrics"
    debug: bool = False
    workers: int = 1
    timeout_keep_alive: int = 600
    proxy_connect_timeout: int = 10
    proxy_read_timeout: int = 300
    proxy_write_timeout: int = 30
    proxy_pool_timeout: int | None = None
    max_connections: int = 100  # Maximum simultaneous connections
    max_keepalive_connections: int = 20  # Keepalive connections
    host: str = "0.0.0.0"
    port: int = 5000
    allowed_origins: list[str] = ["*"]

    # Fake LLM
    fake_llm_host: str = "0.0.0.0"
    fake_llm_port: int = 5001

    # Feature toggles
    proxy_enabled: bool = False
    prompt_hub_enabled: bool = False
    chain_hub_enabled: bool = False
    llm_hub_enabled: bool = False
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

    # Logging
    log_level: str = "INFO"
    log_file: str | None = None
    silence_probes: bool = True
    debug_network_check: bool = True

    # Proxy
    proxy_exclude_headers: str = (
        "host,authorization,cookie,x-forwarded-*,jwt-*,content-length"
    )
    proxy_verify_ssl: bool = True
    proxy_ca_bundle_path: str | None = None
    proxy_target_url: str = (
        "http://localhost:8001"  # Target backend server to forward requests
    )
    # Same meanings as provider auth types (case-insensitive):
    #   OIDC   - token fetched from proxy_oidc_authorization_url: "Authorization: Bearer <token>"
    #   BEARER - static proxy_api_key: "Authorization: Bearer <proxy_api_key>"
    #   APIKEY - static proxy_api_key: "X-API-KEY: <proxy_api_key>"
    #   CERT   - mutual TLS with proxy_api_cert_path / proxy_api_cert_key_path
    #   NONE   - no auth
    proxy_authorization_type: Literal["OIDC", "BEARER", "APIKEY", "CERT", "NONE"] = "OIDC"
    # in case of proxy_authorization_type == "APIKEY" or "BEARER"
    proxy_api_key: str | None = None
    # in case of proxy_authorization_type == "CERT"
    proxy_api_cert_path: str | None = None
    proxy_api_cert_key_path: str | None = None
    # in case of proxy_authorization_type == "OIDC"
    proxy_oidc_authorization_url: str | None = None
    proxy_oidc_credentials: str | None = None

    @field_validator("proxy_authorization_type", mode="before")
    @classmethod
    def _normalize_proxy_authorization_type(cls, v):
        return v.strip().upper() if isinstance(v, str) else v

    @model_validator(mode="after")
    def _migrate_legacy_bearer(self) -> "AppSettings":
        # BEARER used to mean OIDC. Keep old deployments (OIDC URL set, no static
        # key) working instead of silently sending requests without a token.
        if (
            self.proxy_authorization_type == "BEARER"
            and not self.proxy_api_key
            and self.proxy_oidc_authorization_url
        ):
            warnings.warn(
                "proxy_authorization_type=BEARER with proxy_oidc_authorization_url "
                "is deprecated: BEARER now means a static proxy_api_key; use OIDC",
                DeprecationWarning,
                stacklevel=2,
            )
            self.proxy_authorization_type = "OIDC"
        return self

    proxy_pricing_config: str | None = (
        None  # Path to pricing JSON for legacy proxy mode
    )
    proxy_pricing_endpoint: str | None = (
        None  # URL to fetch pricing dynamically (e.g. https://openrouter.ai/api/v1/models_info)
    )
    proxy_pricing_prefix: str = (
        "proxy"  # Provider prefix used when fetching from endpoint
    )
    proxy_pricing_currency: str = "USD"  # Currency when fetching from endpoint

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


# Load settings
settings = AppSettings()

os.environ["LLM_CHAIN_DEFAULT_BASE_URL"] = settings.chain_default_base_url
os.environ["LLM_CHAIN_DEFAULT_MODEL_NAME"] = settings.chain_default_model_name
os.environ["PROMETHEUS_MULTIPROC_DIR"] = settings.metrics_path
