"""OpenAI LLM client."""

from __future__ import annotations

from .base import BaseLLMClient


class OpenAIClient(BaseLLMClient):
    """OpenAI API client."""

    def __init__(self, api_key: str):
        super().__init__(api_key)
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("Install openai: pip install openai")

    async def chat(self, messages: list[dict], model_id: str, temperature: float = 0.7, max_tokens: int = 4096, **kwargs) -> str:
        response = await self.client.chat.completions.create(
            model=model_id, messages=messages, temperature=temperature, max_tokens=max_tokens, **kwargs
        )
        return response.choices[0].message.content or ""

    def supports_tools(self, model_id: str) -> bool:
        return model_id.startswith(("gpt-4", "gpt-3.5-turbo", "o1", "o3"))
