import re
from enum import Enum
from pydantic import BaseModel, Field

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
    # X-Group-Id values allowed to use this provider. An entry matches the exact
    # group_id and its nested ids ("org" matches "org" and "org/user").
    # None or empty: open to everyone.
    allow_list: list[str] | None = None

    def is_group_allowed(self, group_id: str) -> bool:
        if not self.allow_list:
            return True
        return any(group_id == g or group_id.startswith(f"{g}/") for g in self.allow_list)
