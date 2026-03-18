from pydantic import BaseModel, Field
from typing import Any


class HealthCheck(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status (ok, degraded, down)")
    components: dict[str, str] = Field(..., description="Status of individual components")
    version: str = Field(..., description="Service version")
    details: dict[str, Any] | None = Field(None, description="Per-component diagnostic details (only present when degraded)")
