import os
from abc import ABC, abstractmethod
from langchain.chains import load_chain
from app.core.config import settings
from app.core.logging_config import setup_logging
from app.schemas.prompt import PromptVariables
import json

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
    
    async def get_chains(self) -> list[str]:
        """Get a list of all available chains.

        Returns:
            List of chain names
        """
        pass


class StaticChainStore(ChainStore):
    def __init__(self, chains_directory: str):
        """Initialize StaticChainStore with a directory containing chain files.
        
        Args:
            chains_directory: Path to directory containing individual chain JSON files
        """
        self.chains_directory = chains_directory
        self.stored_chains = {}
        self._load_chains()

    def _load_chains(self):
        """Load all chain files from the chains directory."""
        if not os.path.exists(self.chains_directory):
            raise ValueError(f"Chains directory not found: {self.chains_directory}")

        for filename in os.listdir(self.chains_directory):
            if filename.endswith('.json'):
                with open(os.path.join(self.chains_directory, filename), 'r') as f:
                    chain_data = json.load(f)
                chain_path = os.path.join(self.chains_directory, filename)
                chain_name = os.path.splitext(filename)[0]
                chain_info = {
                    "path": chain_path,
                    "description": chain_data.get("metadata", {}).get("description", ""),
                    "category": chain_data.get("metadata", {}).get("category", ""),
                    "input_variables": chain_data.get("prompt", {}).get("input_variables", []),
                    "partial_variables": chain_data.get("prompt", {}).get("partial_variables", {})
                }
                self.stored_chains[chain_name] = chain_info

    async def execute(self, chain_name: str, variables: PromptVariables) -> str:
        if chain_name not in self.stored_chains:
            raise ValueError(f"Chain {chain_name} not found")

        chain = load_chain(self.stored_chains[chain_name]["path"])

        return chain.invoke(variables.root)

    async def get_chains(self) -> list[str]:
        keys_to_keep = {"description", "category", "input_variables", "partial_variables"}
        filtered_data = {
            key: {k: v for k, v in value.items() if k in keys_to_keep}
            for key, value in self.stored_chains.items()
        }
        return filtered_data

chainStore = StaticChainStore(settings.chain_hub_directory)