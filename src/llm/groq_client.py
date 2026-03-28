import json
from groq import AsyncGroq
from loguru import logger


class GroqClient:
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.client = AsyncGroq(api_key=api_key)
        self.model = model

    async def chat(self, system_prompt: str, user_message: str,
                   temperature: float = 0.3, max_tokens: int = 2048) -> str:
        """Send a chat completion request and return the response text."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error("Groq API error: {}", e)
            raise

    async def chat_json(self, system_prompt: str, user_message: str,
                        temperature: float = 0.1, max_tokens: int = 2048) -> dict:
        """Send a request expecting JSON response."""
        try:
            response = await self.client.chat.completions.create(
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
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON from Groq response: {}", e)
            raise
        except Exception as e:
            logger.error("Groq API error: {}", e)
            raise
