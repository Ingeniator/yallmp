from pydantic import RootModel

class PromptVariables(RootModel[dict[str, str]]):
    pass
