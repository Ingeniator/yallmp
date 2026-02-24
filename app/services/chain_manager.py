import os
from abc import ABC, abstractmethod
from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.loading import load_chain
from langchain_community.llms.loading import load_llm_from_config
from app.core.config import settings
from app.core.logging_config import setup_logging
from app.core.security import redact_headers
from app.schemas.prompt import PromptVariables, ChainMetadataForTracking
from app.services.metrics_callback_handler import MetricsCallbackHandler
import json
from fastapi.responses import JSONResponse
from fastapi import HTTPException

logger = setup_logging()

class ChainStore(ABC):
    @abstractmethod
    async def execute(self, chain_name: str, variables: PromptVariables) -> str:
        """Execute a chain by name with the given variables.

        Args:
            chain_name: Name of the chain to execute
            **variables: Variables to format the chain's prompt with

        Returns:
            Result of the chain execution

        Raises:
            ValueError: If chain is not found
        """
        pass
    
    async def get_chains(self) -> dict:
        """Get all available chains.

        Returns:
            Dict of chain names to chain info
        """
        pass

def _redact_response_headers(headers):
    """Redact sensitive values from response headers (dict or string)."""
    if isinstance(headers, dict):
        return redact_headers(headers)
    # If headers came as a raw string / bytes, drop them entirely
    return "[REDACTED]"

def safe_parse_gigachat_exception(e):
    def safe_to_str(value):
        if isinstance(value, bytes):
            return value.decode('utf-8', errors='replace')
        return value
    try:
        url, status_code, content, headers = e.args
    except Exception as parse_error:
        return {
            "url": None,
            "status_code": None,
            "content": None,
            "headers": None,
            "message": str(e),
            "type": type(e).__name__,
            "parse_error": str(parse_error)
        }
    try:
        first_pass = json.loads(safe_to_str(content)) if isinstance(content, str) else safe_to_str(content)
        content_data = json.loads(first_pass)['message'] if isinstance(first_pass, str) else first_pass
        return {
            "url": safe_to_str(url),
            "status_code": safe_to_str(status_code),
            "content": safe_to_str(content),
            "headers": _redact_response_headers(headers),
            "message": content_data
        }
    except (json.JSONDecodeError, TypeError) as e:
        content_data = { "raw": safe_to_str(content) }
        return {
            "url": safe_to_str(url),
            "status_code": safe_to_str(status_code),
            "content": safe_to_str(content),
            "headers": _redact_response_headers(headers),
            "message": content_data or str(e),
            "type": type(e).__name__,
        }

class StaticChainStore(ChainStore):
    def __init__(self, chains_directory: str):
        """Initialize StaticChainStore with a directory containing chain files.
        
        Args:
            chains_directory: Path to directory containing individual chain JSON files
        """
        self.chains_directory = chains_directory
        self.stored_chains = {}
        self._load_chains()
        self.default_available_chat_models = settings.chain_default_available_chat_models

    def _load_chains(self):
        """Load all chain files from the chains directory."""
        if not os.path.exists(self.chains_directory):
            raise ValueError(f"Chains directory not found: {self.chains_directory}")

        for filename in os.listdir(self.chains_directory):
            if filename.endswith('.json'):
                chain_path = os.path.join(self.chains_directory, filename)
                try:
                    with open(chain_path, 'r') as f:
                        chain_data = json.load(f)
                except (json.JSONDecodeError, OSError) as e:
                    logger.error(f"Failed to load chain file {filename}: {e}")
                    continue
                chain_name = os.path.splitext(filename)[0]
                chain_info = {
                    "path": chain_path,
                    "metadata": {
                        "description": chain_data.get("metadata", {}).get("description", ""),
                        "category": chain_data.get("metadata", {}).get("category", "")
                    },
                    "input_variables": chain_data.get("prompt", {}).get("input_variables", []),
                    "partial_variables": chain_data.get("prompt", {}).get("partial_variables", {}),
                    "model": chain_data.get("llm", {}).get("model", {}),
                }
                self.stored_chains[chain_name] = chain_info

    async def execute(self, chain_name: str, variables: PromptVariables, model_name: str | None = None, metadata: ChainMetadataForTracking | None = None) -> str:
        if chain_name not in self.stored_chains:
            logger.warning(f"Chain {chain_name} not found")
            raise HTTPException(status_code=404, detail=f"Chain {chain_name} not found")
        # Gigachat has some serialization issue - Langchain send TypeError when use load_chain_from_config
        # config = await self.read_config(self.stored_chains[chain_name]["path"])
        # chain = load_chain_from_config(config)
        chain = load_chain(self.stored_chains[chain_name]["path"])
        return await self.execute_chain(chain=chain,  
                                        model_name=model_name, 
                                        variables=variables.root, 
                                        metadata=metadata)
        
    async def read_config(self, path: str) -> dict:
        with open(path, "r") as f:
            raw = f.read()
            config = json.loads(os.path.expandvars(raw))
            return config

    async def execute_prompt(self, prompt: str, model_name: str, metadata: ChainMetadataForTracking | None = None) -> str:
        llm = load_llm_from_config(await self.read_config(settings.chain_default_json_file))
        return await self.execute_chain(chain=LLMChain(llm=llm, prompt=PromptTemplate(input_variables=[],template=prompt)), model_name=model_name, metadata=metadata)

    async def get_default_available_chat_models(self, exclude: str | None = None):
        if exclude in self.default_available_chat_models:
            self.default_available_chat_models.remove(exclude)
        return self.default_available_chat_models

    async def execute_chain(self, chain: LLMChain, model_name: str | None = None, variables: dict | None = None, metadata: ChainMetadataForTracking | None = None) -> str:
        variables = variables or {}
        # chain patching
        if not chain.llm.timeout:
            chain.llm.timeout = settings.timeout_keep_alive
        # set base url if it's not defined
        if not chain.llm.base_url and settings.chain_default_base_url:
            chain.llm.base_url = settings.chain_default_base_url
        # set base model if it's not defined
        if not chain.llm.model and settings.chain_default_model_name:
            chain.llm.model = settings.chain_default_model_name
        # override model if needed
        if model_name:
            chain.llm.model = model_name

        if not chain.llm.ca_bundle_file and settings.chain_default_ca_bundle_file:
            chain.llm.ca_bundle_file = settings.chain_default_ca_bundle_file

        if not chain.llm.cert_file and settings.chain_default_cert_file and \
            not chain.llm.key_file and settings.chain_default_key_file:
            chain.llm.cert_file = settings.chain_default_cert_file
            chain.llm.key_file = settings.chain_default_key_file

        if not chain.llm.auth_url and settings.chain_default_auth_url:
            chain.llm.auth_url = settings.chain_default_auth_url
            chain.llm.credentials = settings.chain_default_credentials
            chain.llm.scope = settings.chain_default_scope

        metrics_handler = MetricsCallbackHandler(metadata=metadata)
        callbacks = [metrics_handler]

        if settings.tracing_enabled:
            try:
                from app.services.tracing import get_emitter
                emitter = get_emitter()
                if emitter:
                    cb = emitter.get_langchain_callback(
                        trace_name="chain-execution",
                        metadata={
                            "chain_name": metadata.chain_name if metadata else None,
                            "group_id": metadata.group_id if metadata else None,
                        },
                    )
                    if cb:
                        callbacks.append(cb)
            except Exception as e:
                logger.error("Failed to initialize tracing callback handler", exc_info=e)

        max_fallbacks = len(self.default_available_chat_models)
        for attempt in range(max_fallbacks + 1):
            try:
                return await chain.ainvoke(variables, config={"callbacks": callbacks})
            except Exception as e:
                response_error = safe_parse_gigachat_exception(e)
                error_response = {
                    "error": {
                        "message": response_error["message"] or "Chain execution failed",
                        "details": {
                            "exception": str(e),
                            "exception_type": type(e).__name__
                        }
                    }
                }
                logger.warning("Chain execution failed", details=error_response)
                status_code = response_error.get("status_code")
                message = response_error.get("message", "")
                is_model_not_found = (
                    status_code == 404
                    and isinstance(message, str)
                    and "No such model" in message
                )
                if is_model_not_found:
                    logger.warning(f"404: No such model. {chain.llm.model}")
                    chat_models = await self.get_default_available_chat_models(exclude=chain.llm.model)
                    if not chat_models:
                        return JSONResponse(content={"error": "Chain execution failed", "details": error_response}, status_code=status_code or 500)
                    chain.llm.model = chat_models[0]
                    logger.warning(f"Fallback: Let's try this one - {chain.llm.model}")
                else:
                    return JSONResponse(content={"error": "Chain execution failed", "details": error_response}, status_code=status_code or 500)
        return JSONResponse(content={"error": "No llm models available"}, status_code=404)


    async def get_chains(self, category: str | None = None) -> dict:
        keys_to_keep = {"model", "metadata", "input_variables", "partial_variables"}
        filtered = self.stored_chains
        if category:
            filtered = {key:value for key, value in self.stored_chains.items() if value.get("metadata", {}).get("category") == category}
        filtered_data = {
            key: {k: v for k, v in value.items() if k in keys_to_keep}
            for key, value in filtered.items()
        }
        return filtered_data

chainStore = StaticChainStore(settings.chain_hub_directory)
