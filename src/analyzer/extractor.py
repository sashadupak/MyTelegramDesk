from src.llm.groq_client import GroqClient
from loguru import logger

SYSTEM_PROMPT = """Ты — эксперт по анализу вакансий. Твоя задача — извлечь структурированную информацию из текста поста с вакансией.

Верни JSON со следующими полями:
{
  "is_vacancy": true/false,
  "title": "название позиции или null",
  "company": "название компании или null",
  "location": "локация или null",
  "salary": "зарплата/вилка или null",
  "skills": "ключевые навыки через запятую или null",
  "requirements": "краткие требования или null",
  "remote": true/false/null,
  "contact": "контакт для отклика или null",
  "source_url": "ссылка на вакансию если есть или null"
}

Если текст НЕ является вакансией (новость, реклама, мем и т.д.), верни {"is_vacancy": false} и остальные поля null.
Отвечай ТОЛЬКО валидным JSON без markdown."""


class VacancyExtractor:
    def __init__(self, llm: GroqClient):
        self.llm = llm

    async def extract(self, text: str) -> dict | None:
        """
        Extract structured vacancy data from raw text.
        Returns None if text is not a vacancy.
        """
        try:
            result = await self.llm.chat_json(SYSTEM_PROMPT, text)
            if not result.get("is_vacancy"):
                return None
            return result
        except Exception as e:
            logger.error("Extraction failed: {}", e)
            return None
