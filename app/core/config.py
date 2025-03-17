from pydantic_settings import BaseSettings, SettingsConfigDict

class AppSettings(BaseSettings):
    app_name: str = "LLM-Proxy"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # Fake LLM
    fake_llm_host: str = "0.0.0.0"
    fake_llm_port: int = 8001

    # Logging
    log_level: str = "INFO"
    log_file: str | None = None

    # LLM
    llm_api_base_url: str = "http://localhost:8001"
    llm_authorization_type: str = "BEARER"
    # in case of llm_authorization_type == "APIKEY"
    llm_api_key: str | None = None
    # in case of llm_authorization_type == "CERT"
    llm_client_cert_file: str | None = None
    llm_client_key_file: str | None = None
    # in case of llm_authorization_type == "BEARER"
    llm_oidc_authorization_url: str | None = None
    llm_oidc_client_id: str | None = None
    llm_oidc_client_secret: str | None = None

    # Proxy
    proxy_target_url: str = "http://localhost:8001"  # Target backend server to forward requests
    proxy_max_retries: int = 5  # Number of retries before circuit breaker
    proxy_base_delay: float = 0.5  # Base delay in seconds
    proxy_backoff_factor: float = 2.0  # Exponential backoff multiplier
    # Circuit breaker configuration
    proxy_failure_threshold: int = 5  # Number of failures before tripping
    proxy_recovery_time: int = 30  # Cooldown period (seconds)
    proxy_window_size: int = 60  # Sliding window size in seconds
    
    model_config = SettingsConfigDict(env_prefix="LLM_PROXY_", env_file=".env", env_file_encoding="utf-8")

# Load settings
settings = AppSettings()
