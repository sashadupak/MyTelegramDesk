import asyncio
import json
import re
from openai import AsyncOpenAI
from loguru import logger


class GroqClient:
    """Universal LLM client supporting Cerebras, Groq, and any OpenAI-compatible API."""

    PROVIDERS = {
        "cerebras": {
            "base_url": "https://api.cerebras.ai/v1",
            "default_model": "llama3.1-8b",
        },
        "groq": {
            "base_url": "https://api.groq.com/openai/v1",
            "default_model": "llama-3.3-70b-versatile",
        },
    }

    def __init__(self, api_key: str, model: str = None, provider: str = "cerebras",
                 fallback_api_key: str = None, fallback_provider: str = None,
                 fallback_model: str = None):
        # Primary provider
        self.provider = provider
        cfg = self.PROVIDERS.get(provider, {})
        self.model = model or cfg.get("default_model", "llama3.1-8b")
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=cfg.get("base_url", "https://api.cerebras.ai/v1"),
        )

        # Fallback provider (optional)
        self.fallback_client = None
        self.fallback_model = None
        if fallback_api_key and fallback_provider:
            fb_cfg = self.PROVIDERS.get(fallback_provider, {})
            self.fallback_model = fallback_model or fb_cfg.get("default_model")
            self.fallback_client = AsyncOpenAI(
                api_key=fallback_api_key,
                base_url=fb_cfg.get("base_url"),
            )
            logger.info("LLM: {} ({}) + fallback {} ({})",
                        provider, self.model, fallback_provider, self.fallback_model)
        else:
            logger.info("LLM: {} ({})", provider, self.model)

    @staticmethod
    def _parse_retry_seconds(error_msg: str) -> float:
        """Extract wait time from rate limit error."""
        match = re.search(r"try again in (\d+)m([\d.]+)s", str(error_msg))
        if match:
            return int(match.group(1)) * 60 + float(match.group(2)) + 5
        match = re.search(r"try again in ([\d.]+)s", str(error_msg))
        if match:
            return float(match.group(1)) + 5
        return 60

    async def _call_with_retry(self, **kwargs) -> object:
        """Call API with retry on rate limit, then fallback to secondary provider."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return await self.client.chat.completions.create(**kwargs)
            except Exception as e:
                err = str(e)
                if "429" in err or "rate_limit" in err.lower():
                    if self.fallback_client and attempt == 0:
                        logger.warning("{} rate limit — switching to fallback", self.provider)
                        try:
                            fb_kwargs = {**kwargs, "model": self.fallback_model}
                            return await self.fallback_client.chat.completions.create(**fb_kwargs)
                        except Exception as fb_e:
                            logger.warning("Fallback also failed: {}", fb_e)

                    wait = self._parse_retry_seconds(err)
                    logger.warning("Rate limit. Waiting {:.0f}s (attempt {}/{})...",
                                   wait, attempt + 1, max_retries)
                    await asyncio.sleep(wait)
                else:
                    logger.error("LLM API error: {}", e)
                    raise
        raise RuntimeError("LLM rate limit: max retries exceeded")

    async def chat(self, system_prompt: str, user_message: str,
                   temperature: float = 0.3, max_tokens: int = 2048) -> str:
        response = await self._call_with_retry(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    async def chat_json(self, system_prompt: str, user_message: str,
                        temperature: float = 0.1, max_tokens: int = 2048) -> dict:
        response = await self._call_with_retry(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON: {}", e)
            raise
