from src.llm.groq_client import GroqClient
from loguru import logger

SYSTEM_PROMPT = """Ты — карьерный консультант, который помогает составлять персонализированные сопроводительные письма.

Тебе даны:
1. Профиль кандидата (навыки, опыт, интро "о себе")
2. Информация о вакансии
3. Досье на компанию (talking_points — зацепки)

Твоя задача — адаптировать интро "о себе" под конкретную вакансию и компанию:
- Сделай акцент на РЕЛЕВАНТНЫХ навыках и опыте для этой конкретной позиции
- Используй talking_points чтобы показать осведомлённость о компании
- Покажи мотивацию — ПОЧЕМУ именно эта компания и позиция интересны
- Пиши от первого лица, естественно, без канцеляризмов
- Длина: 4-6 предложений

Верни JSON:
{
  "cover_letter": "Текст адаптированного интро",
  "highlights": ["ключевой акцент 1", "ключевой акцент 2"]
}

Отвечай ТОЛЬКО валидным JSON."""


class CoverLetterGenerator:
    def __init__(self, llm: GroqClient, profile: dict):
        self.llm = llm
        self.profile = profile

    async def generate(self, vacancy: dict, company_info: dict,
                       match_result: dict) -> dict:
        """
        Generate a personalized cover letter based on vacancy,
        company research, and match analysis.
        """
        p = self.profile
        user_msg = (
            f"=== ПРОФИЛЬ КАНДИДАТА ===\n"
            f"Имя: {p.get('name')}\n"
            f"Текущая должность: {p.get('title')}\n"
            f"Интро: {p.get('intro')}\n"
            f"Навыки: {', '.join(p.get('skills', []))}\n"
            f"Опыт: {p.get('experience_years')} лет\n\n"
            f"=== ВАКАНСИЯ ===\n"
            f"Позиция: {vacancy.get('title', 'N/A')}\n"
            f"Компания: {vacancy.get('company', 'N/A')}\n"
            f"Требования: {vacancy.get('requirements', 'N/A')}\n"
            f"Навыки: {vacancy.get('skills', 'N/A')}\n\n"
            f"=== ДОСЬЕ НА КОМПАНИЮ ===\n"
            f"Описание: {company_info.get('summary', 'N/A')}\n"
            f"Отрасль: {company_info.get('industry', 'N/A')}\n"
            f"Культура: {company_info.get('culture_hints', 'N/A')}\n"
            f"Зацепки: {', '.join(company_info.get('talking_points', []))}\n\n"
            f"=== АНАЛИЗ СОВПАДЕНИЯ ===\n"
            f"Сильные стороны: {', '.join(match_result.get('strengths', []))}\n"
            f"Пробелы: {', '.join(match_result.get('gaps', []))}"
        )

        try:
            result = await self.llm.chat_json(SYSTEM_PROMPT, user_msg)
            return result
        except Exception as e:
            logger.error("Cover letter generation failed: {}", e)
            return {
                "cover_letter": p.get("intro", ""),
                "highlights": [],
            }
