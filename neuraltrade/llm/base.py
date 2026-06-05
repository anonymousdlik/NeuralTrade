"""Base class for LLM provider clients."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        model_id: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> str:
        """Send a chat completion request."""
        ...

    @abstractmethod
    def supports_tools(self, model_id: str) -> bool:
        """Check if the model supports tool/function calling."""
        ...

    async def close(self):
        """Close the client connection."""
        pass
