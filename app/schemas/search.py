from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.provider import AuthConfig, TimeoutConfig


class SearchProviderConfig(BaseModel):
    name: str
    type: str  # "tavily" | "exa" | "brave"
    auth: AuthConfig = Field(default_factory=AuthConfig)
    base_url: str
    default: bool = False
    timeout: TimeoutConfig = Field(default_factory=TimeoutConfig)
    cost_per_search: float = 0.0
    failure_threshold: int = 0
    recovery_time: int = 30
    window_size: int = 60


class SearchRequest(BaseModel):
    query: str
    provider: str | None = None
    num_results: int = Field(default=5, ge=1, le=20)
    search_depth: str = "basic"  # "basic" | "advanced"
    include_raw_content: bool = False
    include_domains: list[str] = Field(default_factory=list)
    exclude_domains: list[str] = Field(default_factory=list)


class SearchResult(BaseModel):
    url: str
    title: str
    content: str
    score: float | None = None
    raw_content: str | None = None


class SearchResponse(BaseModel):
    results: list[SearchResult]
    provider: str
    query: str
    answer: str | None = None
