import json
import os
from abc import ABC, abstractmethod
from langchain.prompts import PromptTemplate
from app.core.config import settings
from app.schemas.prompt import PromptVariables
from app.core.logging_config import setup_logging
from fastapi import HTTPException
logger = setup_logging()

class PromptStore(ABC):
    @abstractmethod
    async def format_prompt(self, prompt_name: str, variables: PromptVariables) -> str:
        """Get a prompt by name and format it with the given variables.

        Args:
            prompt_name: Name of the prompt to retrieve
            **variables: Variables to format the prompt with

        Returns:
            Formatted prompt string

        Raises:
            ValueError: If prompt is not found
        """
        pass
    
    async def get_prompts(self) -> list[str]:
        """Get a list of all available prompts.
        Returns:
            List of prompt names
        """
        pass


class StaticPromptStore(PromptStore):
    def __init__(self, prompts_directory: str):
        """Initialize StaticPromptStore with a directory containing prompt files.
        
        Args:
            prompts_directory: Path to directory containing individual prompt JSON files
        """
        self.prompts_directory = prompts_directory
        self.stored_prompts = {}
        self._load_prompts()

    def _load_prompts(self):
        """Load all prompt files from the prompts directory."""
        if not os.path.exists(self.prompts_directory):
            raise ValueError(f"Prompts directory not found: {self.prompts_directory}")

        for filename in os.listdir(self.prompts_directory):
            if filename.endswith('.json'):
                prompt_path = os.path.join(self.prompts_directory, filename)
                with open(prompt_path, 'r') as f:
                    prompt_data = json.load(f)
                    prompt_name = os.path.splitext(filename)[0]
                    self.stored_prompts[prompt_name] = prompt_data

    async def format_prompt(self, prompt_name: str, variables: PromptVariables) -> str:
        if prompt_name not in self.stored_prompts:
            raise ValueError(f"Prompt {prompt_name} not found")

        prompt_info = self.stored_prompts[prompt_name]
        try:
            loaded_template = PromptTemplate.parse_obj(prompt_info)
        except Exception as e:
            logger.error(f"Failed to parse prompt {prompt_name}: {e}")
            raise

        try:
            # Merge user-provided data with default partial variables
            variables = {**loaded_template.partial_variables, **variables.root}

            # Ensure all required variables are present
            missing_keys = set(loaded_template.input_variables) - set(variables.keys())
            if missing_keys:
                raise KeyError(f"Missing required keys: {', '.join(missing_keys)}")

            return loaded_template.format(**variables)
        except KeyError as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def get_prompts(self) -> list[str]:
        keys_to_keep = {"description", "category", "input_variables", "partial_variables"}
        filtered_data = {
    key: {k: v for k, v in value.items() if k in keys_to_keep}
    for key, value in self.stored_prompts.items()
}
        return filtered_data

promptStore = StaticPromptStore(settings.prompt_hub_directory)