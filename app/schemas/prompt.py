from pydantic import RootModel, BaseModel, Field
from enum import Enum

class PromptVariables(RootModel[dict[str, str]]):
    pass

class ChainType(str, Enum):
    unknown = "unknown"
    prompt = "prompt"
    chain = "chain"

class ChainMetadataForTracking(BaseModel):
    """Metadata for a chain"""
    chain_type: ChainType
    chain_name: str = Field(default="unknown", description="Chain/prompt name used to track metrics")
    group_id: str = Field(default="unknown", description="Group id used to track metrics")
