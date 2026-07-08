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


class AliasEntry(BaseModel):
    target: str
    fallback: str | None = None


class PricingInfo(BaseModel):
    input_cost_per_token: float = 0.0
    output_cost_per_token: float = 0.0
    # Per-character cost for TTS-style models (e.g. OpenAI tts-1).
    cost_per_character: float | None = None
    # Per-image cost for image-generation models, keyed by "{size}:{quality}"
    # (falls back to plain "{size}" for models like dall-e-2 with no quality tiers).
    image_cost: dict[str, float] | None = None


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
    currency: str = "USD"
    pricing_endpoint: str | None = None
    pricing: dict[str, PricingInfo] | None = None
