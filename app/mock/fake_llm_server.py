from fastapi import Depends, FastAPI
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

    # Chat Completions
    @app.post("/v1/chat/completions")
    async def chat(llm: BaseChatModel = Depends(get_fake_llm)):
        return llm.invoke("Hello, world!").response_metadata

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
