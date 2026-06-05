"""Google Gemini LLM client."""

from __future__ import annotations

from .base import BaseLLMClient


class GoogleClient(BaseLLMClient):
    """Google Gemini API client."""

    def __init__(self, api_key: str):
        super().__init__(api_key)
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.genai = genai
        except ImportError:
            raise ImportError("Install google-generativeai: pip install google-generativeai")

    async def chat(self, messages: list[dict], model_id: str, temperature: float = 0.7, max_tokens: int = 4096, **kwargs) -> str:
        model = self.genai.GenerativeModel(model_id)
        # Convert messages to Gemini format
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        response = await model.generate_content_async(prompt, generation_config=self.genai.types.GenerationConfig(
            temperature=temperature, max_output_tokens=max_tokens
        ))
        return response.text or ""

    def supports_tools(self, model_id: str) -> bool:
        return True
