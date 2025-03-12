from pydantic_settings import BaseSettings, SettingsConfigDict

class AppSettings(BaseSettings):
    app_name: str = "LLM-Proxy"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # Logging
    log_level: str = "INFO"
    log_file: str | None = None

    model_config = SettingsConfigDict(env_prefix="LLM_PROXY_", env_file=".env", env_file_encoding="utf-8")

# Load settings
settings = AppSettings()
