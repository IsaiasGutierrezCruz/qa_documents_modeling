from abc import ABC, abstractmethod
from typing import Any, Literal

from llama_index.llms.openai import OpenAI


class LLMService(ABC):
    @abstractmethod
    def chat(self, messages: list[dict[str, str]]) -> dict:
        """Handles a chat conversation."""
        pass

    @abstractmethod
    def get_model(self) -> Any:
        """Handles a chat conversation."""
        pass


class OpenAILLMServiceLLamaIndexAdapter(LLMService):
    def __init__(
        self, model: Literal["gpt-4o", "gpt-3.5-turbo"], temperature: float,
    ) -> None:
        self.model = OpenAI(model=model, temperature=temperature)

    def chat(self, messages: list[dict[str, str]]) -> dict:
        return self.model.chat(messages)

    def get_model(self) -> Any:
        return self.model


def get_llm_service(provider: Literal["openai"]) -> LLMService:
    if provider == "openai":
        return OpenAILLMServiceLLamaIndexAdapter(model="gpt-4o", temperature=0.3)
    else:
        msg = f"Unsupported LLM provider: {provider}"
        raise ValueError(msg)
