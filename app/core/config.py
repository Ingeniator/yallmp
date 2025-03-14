from pydantic_settings import BaseSettings, SettingsConfigDict

class AppSettings(BaseSettings):
    app_name: str = "LLM-Proxy"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # Logging
    log_level: str = "INFO"
    log_file: str | None = None

    # LLM
    llm_api_base_url: str = "http://localhost:8000/v1"
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

    model_config = SettingsConfigDict(env_prefix="LLM_PROXY_", env_file=".env", env_file_encoding="utf-8")

# Load settings
settings = AppSettings()
