from pydantic import BaseModel, Field

class HealthCheck(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status (ok, degraded, down)")
    components: dict[str, str] = Field(..., description="Status of individual components")
    version: str = Field(..., description="Service version")
