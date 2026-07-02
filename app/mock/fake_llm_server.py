import asyncio
import json
import time
from fastapi import Depends, FastAPI, Request
from fastapi.responses import StreamingResponse
from app.mock.fake_llm import get_fake_llm
from app.core.config import settings
from app.core.logging_config import setup_logging
from app.middlewares.logging_middleware import LoggingMiddleware
from langchain.chat_models.base import BaseChatModel

logger = setup_logging()

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="Fake LLM Service", debug=settings.debug)

    @app.on_event("startup")
    async def startup_event():
        logger.info("Fake LLM startup...")

    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Fake LLM shutdown...")

    # Add Logging Middleware
    app.add_middleware(LoggingMiddleware)

    @app.get("/status")
    async def status():
        return {"status": "Fake LLM is running"}

    # Models
    @app.get("/v1/models")
    async def models():
        return {
                    "object": "list",
                    "data": [
                        {
                        "id": "fake-model-id-0",
                        "object": "model",
                        "created": 1686935002,
                        "owned_by": "fakerai"
                        },
                        {
                        "id": "fake-model-id-1",
                        "object": "model",
                        "created": 1686935002,
                        "owned_by": "fakerai"
                        }
                    ],
                }

    _STREAMING_WORDS = ["Hello", ",", " I", " am", " a", " fake", " streaming", " LLM", "."]

    # Chat Completions — supports both sync and streaming (stream: true)
    @app.post("/v1/chat/completions")
    async def chat(req: Request, llm: BaseChatModel = Depends(get_fake_llm)):
        try:
            body = await req.json()
        except Exception:
            body = {}

        if not body.get("stream"):
            return llm.invoke("Hello, world!").response_metadata

        model = body.get("model", "fake-model-id-0")
        completion_id = f"chatcmpl-fake-{int(time.time())}"

        async def _sse_generator():
            for i, word in enumerate(_STREAMING_WORDS):
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": word}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0)

            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": len(_STREAMING_WORDS), "total_tokens": 10 + len(_STREAMING_WORDS)},
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_sse_generator(), media_type="text/event-stream")

    # Embeddings
    @app.post("/v1/embeddings")
    async def embeddings(llm: BaseChatModel = Depends(get_fake_llm)):
        return {
            "object": "list",
            "data": [
                {
                "object": "embedding",
                "embedding": [
                    0.0023064255,
                    -0.009327292,
                    -0.0028842222,
                ],
                "index": 0
                }
            ],
            "model": "fake-embedding-001",
            "usage": {
                "prompt_tokens": 8,
                "total_tokens": 8
            }
        }

    # Fine-tuning
    
    @app.post("/v1/fine_tuning/jobs")
    async def create_fine_tuning_jobs():
        return {
            "object": "fine_tuning.job",
            "id": "ftjob-abc123",
            "model": "fake-llm",
            "created_at": 1721764800,
            "fine_tuned_model": None,
            "organization_id": "org-123",
            "result_files": [],
            "status": "queued",
            "validation_file": None,
            "training_file": "file-abc123",
            "method": {
                "type": "supervised",
                "supervised": {
                "hyperparameters": {
                    "batch_size": "auto",
                    "learning_rate_multiplier": "auto",
                    "n_epochs": "auto",
                }
                }
            },
            "metadata": None
        }

    @app.get("/v1/fine_tuning/jobs")
    async def get_fine_tuning_jobs():
        return {
                "object": "list",
                "data": [
                    {
                    "object": "fine_tuning.job",
                    "id": "ftjob-abc123",
                    "model": "fakellm",
                    "created_at": 1721764800,
                    "fine_tuned_model": None,
                    "organization_id": "org-123",
                    "result_files": [],
                    "status": "queued",
                    "validation_file": None,
                    "training_file": "file-abc123",
                    "metadata": {
                        "key": "value"
                    }
                    },
                ], "has_more": True
        }

    @app.delete("/v1/models/{model}")
    async def delete_fine_tuning_model(model: str):
        return {
            "id": f"ft:model-name:{model}",
            "object": "model",
            "deleted": True
        }

    return app
