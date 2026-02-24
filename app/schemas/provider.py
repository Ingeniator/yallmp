from enum import Enum
from pydantic import BaseModel, Field


class AuthType(str, Enum):
    BEARER = "BEARER"
    APIKEY = "APIKEY"
    CERT = "CERT"
    NONE = "NONE"


class AuthConfig(BaseModel):
    type: AuthType = AuthType.NONE
    oidc_url: str | None = None
    credentials: str | None = None
    scope: str | None = None
    api_key: str | None = None
    cert_path: str | None = None
    cert_key_path: str | None = None


class TimeoutConfig(BaseModel):
    connect: int = 10
    read: int = 300
    write: int = 30
    pool: int | None = None


class LlmProviderConfig(BaseModel):
    prefix: str
    base_url: str
    auth: AuthConfig = Field(default_factory=AuthConfig)
    models: list[str] = Field(default_factory=list)
    verify_ssl: bool = True
    timeout: TimeoutConfig = Field(default_factory=TimeoutConfig)
    failure_threshold: int = 0
    recovery_time: int = 30
    window_size: int = 60
    max_retries: int = 5
    base_delay: float = 0.5
    backoff_factor: float = 2.0
