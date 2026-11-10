import re
from enum import Enum
from pydantic import BaseModel, Field, model_validator

HEADER_PLACEHOLDER = re.compile(r"\$\{headers\.([A-Za-z0-9_-]+)\}")
DEFAULT_PASSTHROUGH_HEADERS = {"X-API-KEY": "${headers.X-API-KEY}"}


class AuthType(str, Enum):
    # Static token: "Authorization: Bearer <api_key>"
    BEARER = "BEARER"
    # Token fetched from oidc_url with credentials/scope: "Authorization: Bearer <token>"
    OIDC = "OIDC"
    # Static key: "X-API-KEY: <api_key>"
    APIKEY = "APIKEY"
    # Mutual TLS client certificate (cert_path, cert_key_path)
    CERT = "CERT"
    NONE = "NONE"
    # Build provider auth headers from the client's request headers (see AuthConfig.headers)
    PASSTHROUGH = "PASSTHROUGH"


class AuthConfig(BaseModel):
    type: AuthType = AuthType.NONE
    # OIDC only
    oidc_url: str | None = None
    credentials: str | None = None
    scope: str | None = None
    # BEARER and APIKEY
    api_key: str | None = None
    # CERT only
    cert_path: str | None = None
    cert_key_path: str | None = None
    # PASSTHROUGH only: upstream header -> template, e.g.
    # {"Authorization": "Bearer ${headers.X-API-KEY}"}.
    # Default: {"X-API-KEY": "${headers.X-API-KEY}"}
    headers: dict[str, str] | None = None

    @model_validator(mode="after")
    def _check_required_fields(self) -> "AuthConfig":
        if self.type == AuthType.BEARER and not self.api_key:
            hint = (
                "; for a token fetched from oidc_url use type OIDC"
                if self.oidc_url
                else ""
            )
            raise ValueError(f"auth type BEARER requires api_key{hint}")
        if self.type == AuthType.OIDC and not self.oidc_url:
            raise ValueError("auth type OIDC requires oidc_url")
        if self.type == AuthType.APIKEY and not self.api_key:
            raise ValueError("auth type APIKEY requires api_key")
        return self

    def passthrough_templates(self) -> dict[str, str]:
        return self.headers or DEFAULT_PASSTHROUGH_HEADERS

    def passthrough_source_headers(self) -> set[str]:
        """Lower-cased names of client headers referenced by the PASSTHROUGH templates."""
        return {
            name.lower()
            for template in self.passthrough_templates().values()
            for name in HEADER_PLACEHOLDER.findall(template)
        }


class TimeoutConfig(BaseModel):
    connect: int = 10
    read: int = 300
    write: int = 30
    pool: int | None = None


class AliasEntry(BaseModel):
    target: str
    fallback: str | None = None


class PricingInfo(BaseModel):
    input_cost_per_token: float
    output_cost_per_token: float


class LlmProviderConfig(BaseModel):
    prefix: str
    base_url: str
    # Receives models that match no prefix and no provider's ``models`` list.
    # At most one provider should set it; the first one loaded wins.
    default: bool = False
    auth: AuthConfig = Field(default_factory=AuthConfig)
    models: list[str] = Field(default_factory=list)
    verify_ssl: bool = True
    timeout: TimeoutConfig = Field(default_factory=TimeoutConfig)
    # Circuit breaker; None uses the global proxy_failure_threshold /
    # proxy_recovery_time / proxy_window_size settings.
    failure_threshold: int | None = None
    recovery_time: int | None = None
    window_size: int | None = None
    max_retries: int = 5
    base_delay: float = 0.5
    backoff_factor: float = 2.0
    currency: str = "USD"
    pricing_endpoint: str | None = None
    pricing: dict[str, PricingInfo] | None = None
    # X-Group-Id values allowed to use this provider. An entry matches the exact
    # group_id and its nested ids ("org" matches "org" and "org/user").
    # None or empty: open to everyone.
    allow_list: list[str] | None = None

    def is_group_allowed(self, group_id: str) -> bool:
        if not self.allow_list:
            return True
        return any(group_id == g or group_id.startswith(f"{g}/") for g in self.allow_list)
