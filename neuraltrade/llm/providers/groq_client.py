"""Groq LLM client — ultra-fast inference."""

from __future__ import annotations

from .base import BaseLLMClient


class GroqClient(BaseLLMClient):
    """Groq API client."""

    def __init__(self, api_key: str):
        super().__init__(api_key)
        try:
            from groq import AsyncGroq
            self.client = AsyncGroq(api_key=api_key)
        except ImportError:
            raise ImportError("Install groq: pip install groq")

    async def chat(self, messages: list[dict], model_id: str, temperature: float = 0.7, max_tokens: int = 4096, **kwargs) -> str:
        response = await self.client.chat.completions.create(
            model=model_id, messages=messages, temperature=temperature, max_tokens=max_tokens, **kwargs
        )
        return response.choices[0].message.content or ""

    def supports_tools(self, model_id: str) -> bool:
        return True
