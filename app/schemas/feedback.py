from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    request_id: str = Field(..., description="X-Request-ID sent with the original LLM call")
    score: float = Field(..., description="Numeric score (e.g. 1/-1 for thumbs, 1-5 for stars, 0.0-1.0 normalised)")
    name: str = Field(default="user_feedback", description="Score name in Langfuse")
    comment: str | None = Field(default=None, description="Optional free-text explanation")


class FeedbackResponse(BaseModel):
    status: str
    trace_id: str = Field(..., description="Langfuse trace ID the score was attached to")
