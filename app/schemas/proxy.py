from typing import Any
from pydantic import BaseModel, Field


class ProxyError(BaseModel):
    """Error response model for proxy requests."""
    status_code: int = Field(..., description="HTTP status code")
    message: str = Field(..., description="Error message")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional error details")
