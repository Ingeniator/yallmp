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
    proxy_pool_timeout: int|None = None
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

    # LLM Hub
    llm_hub_directory: str = "data/llm_hub"

    # Tracing
    tracing_enabled: bool = False
    tracing_log_io: bool = True  # True = log messages+response content; False = metadata only
    tracing_backend: str = "langfuse"  # selects which TraceEmitter to instantiate

    # Logging
    log_level: str = "INFO"
    log_file: str | None = None


    # Proxy
    proxy_exclude_headers: str = "host,authorization,cookie,x-forwarded-*,jwt-*"
    proxy_verify_ssl: bool = True
    proxy_ca_bundle_path: str | None = None
    proxy_target_url: str = "http://localhost:8001"  # Target backend server to forward requests
    proxy_authorization_type: str = "BEARER"
    # in case of proxy_authorization_type == "APIKEY"
    proxy_api_key: str | None = None
    # in case of proxy_authorization_type == "CERT"
    proxy_api_cert_path: str | None = None
    proxy_api_cert_key_path: str | None = None
    # in case of proxy_authorization_type == "BEARER"
    proxy_oidc_authorization_url: str | None = None
    proxy_oidc_credentials: str | None = None

    proxy_max_retries: int = 5  # Number of retries before circuit breaker
    proxy_base_delay: float = 0.5  # Base delay in seconds
    proxy_backoff_factor: float = 2.0  # Exponential backoff multiplier
    # Circuit breaker configuration
    proxy_failure_threshold: int = 0  # 0 if circuit breaker disabled. Number of failures before tripping
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
    chain_default_available_chat_models: list[str] = ["Gigachat:latest", "GigaChat-Pro", "GigaChat-Max", "Gigachat-2:latest", "DeepSeek-R1"]
    chain_default_json_file: str = "./data/llm_hub/langchain/default.json"

    version: str = "0.1.0"
    
    model_config = SettingsConfigDict(env_prefix="LLM_", env_file=".env", env_file_encoding="utf-8", extra='allow')

# Load settings
settings = AppSettings()

os.environ["LLM_CHAIN_DEFAULT_BASE_URL"] = settings.chain_default_base_url
os.environ["LLM_CHAIN_DEFAULT_MODEL_NAME"] = settings.chain_default_model_name
os.environ["PROMETHEUS_MULTIPROC_DIR"] = settings.metrics_path
