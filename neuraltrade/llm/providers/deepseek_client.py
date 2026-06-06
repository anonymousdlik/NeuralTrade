"""DeepSeek LLM client."""

from __future__ import annotations

from .base import BaseLLMClient


class DeepSeekClient(BaseLLMClient):
    """DeepSeek API client (OpenAI-compatible)."""

    def __init__(self, api_key: str):
        super().__init__(api_key)
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com",
            )
        except ImportError:
            raise ImportError("Install openai: pip install openai")

    async def chat(self, messages: list[dict], model_id: str, temperature: float = 0.7, max_tokens: int = 4096, **kwargs) -> str:
        response = await self.client.chat.completions.create(
            model=model_id, messages=messages, temperature=temperature, max_tokens=max_tokens, **kwargs
        )
        return response.choices[0].message.content or ""

    def supports_tools(self, model_id: str) -> bool:
        return True
