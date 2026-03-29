import asyncio
import json
import re
from groq import AsyncGroq
from loguru import logger


class GroqClient:
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.client = AsyncGroq(api_key=api_key)
        self.model = model

    @staticmethod
    def _parse_retry_seconds(error_msg: str) -> float:
        """Extract wait time from Groq 429 error message."""
        match = re.search(r"try again in (\d+)m([\d.]+)s", str(error_msg))
        if match:
            return int(match.group(1)) * 60 + float(match.group(2)) + 5
        match = re.search(r"try again in ([\d.]+)s", str(error_msg))
        if match:
            return float(match.group(1)) + 5
        return 60  # default 1 min

    async def _call_with_retry(self, **kwargs) -> object:
        """Call Groq API with automatic retry on rate limit."""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                return await self.client.chat.completions.create(**kwargs)
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    wait = self._parse_retry_seconds(str(e))
                    logger.warning("Rate limit hit. Waiting {:.0f}s (attempt {}/{})...",
                                   wait, attempt + 1, max_retries)
                    await asyncio.sleep(wait)
                else:
                    logger.error("Groq API error: {}", e)
                    raise
        raise RuntimeError("Groq rate limit: max retries exceeded")

    async def chat(self, system_prompt: str, user_message: str,
                   temperature: float = 0.3, max_tokens: int = 2048) -> str:
        """Send a chat completion request and return the response text."""
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
        """Send a request expecting JSON response."""
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
            logger.error("Failed to parse JSON from Groq response: {}", e)
            raise
