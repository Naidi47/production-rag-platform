import asyncio
from typing import Any

import httpx
import tiktoken

from src.config import config


class LLMClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.base_url = (base_url or config.LLM_BASE_URL).rstrip("/") + "/"
        self.api_key = api_key if api_key is not None else config.LLM_API_KEY
        self.model = model or config.LLM_MODEL
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=config.LLM_TIMEOUT_SECONDS,
        )
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_retries: int | None = None,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("LLM_API_KEY is not configured")

        retries = config.LLM_MAX_RETRIES if max_retries is None else max_retries
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": config.LLM_MAX_TOKENS,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                response = await self.client.post("chat/completions", json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("LLM returned an empty response")
                return content.strip()
            except (httpx.HTTPStatusError, httpx.RequestError, KeyError, IndexError, TypeError) as exc:
                last_error = exc
                if attempt < retries:
                    await asyncio.sleep(min(2**attempt, 8))
        raise RuntimeError(
            f"LLM request failed after {retries + 1} attempts: {last_error}"
        ) from last_error

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    async def close(self) -> None:
        await self.client.aclose()
