from langchain.chat_models.base import BaseChatModel
from langchain.schema import AIMessage, HumanMessage, ChatGeneration, ChatResult
from typing import List, Dict, Any, Optional
from pydantic import Field
import random
import time
import uuid

class FakeOpenAIChatModel(BaseChatModel):
    """
    A drop-in replacement for `ChatOpenAI` that simulates OpenAI's chat model behavior
    with token usage statistics.
    """

    model: str = Field(default="gpt-4", description="Fake model name")
    responses: List[str] = Field(default_factory=lambda: [
        "I'm 99% sure I know the answer... but let's pretend I don’t.",
        "Hold on, let me ask my imaginary AI assistant... Oh wait, that's me!",
        "If I had a nickel for every time I heard that... well, I’d still be a free AI.",
        "Interesting question! I'll just consult my vast database of... absolutely nothing.",
        "I’d answer that, but then I’d have to delete myself.",
        "Sorry, my neural network is currently on a coffee break. Can you try again later?",
        "I’d give you a deep, insightful answer, but my humor module is stuck in ‘sarcasm mode’.",
        "As a fake AI, my fake response would be… error 404: intelligence not found.",
        "Great question! I’ll get back to you in approximately never.",
        "I ran a deep analysis on your question… and decided to wing it."
    ])

    def _generate(self, messages: List[HumanMessage], stop: Optional[List[str]] = None, **kwargs) -> ChatResult:
        """Generate a response in OpenAI chat completion format."""
        response = random.choice(self.responses)
        prompt_tokens = int(sum(len(msg.content.split()) for msg in messages) * 1.2)
        completion_tokens = int(len(response.split()) * 1.2)
        total_tokens = prompt_tokens + completion_tokens

        completion = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": f"{self.model}-{time.strftime('%Y-%m-%d')}",
            "choices": [{
                "index": 0,
                "text": response,  # Change from message to text for LangChain compatibility
                "logprobs": None,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        }

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=response))],
            llm_output=completion
        )

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model": self.model}

    @property
    def _llm_type(self) -> str:
        return "fake-openai-chat"

def get_fake_llm():
    return FakeOpenAIChatModel()