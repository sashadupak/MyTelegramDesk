import pdfplumber
from pathlib import Path
from src.llm.groq_client import GroqClient
from loguru import logger

SYSTEM_PROMPT = """Ты — строгий рекрутер-аналитик. Тебе дан профиль кандидата и вакансия.
Оцени РЕАЛИСТИЧНО, насколько кандидат КОНКУРЕНТОСПОСОБЕН на эту вакансию.

ПРАВИЛА ОЦЕНКИ (строго соблюдай):

1. РОЛЬ должна совпадать с preferred_roles кандидата. Если вакансия на чистого разработчика (Python Developer, Backend Developer, Frontend Developer, Java Developer и т.д.), а кандидат — Product Manager / Innovation Lead, ставь score ≤ 30.

2. ГРЕЙД: если вакансия требует 5+ лет чистого опыта в конкретном стеке (например, backend Python 5+ лет), а у кандидата основной опыт в продуктовом менеджменте и робототехнике — ставь score ≤ 40.

3. НАВЫКИ: считай только ПРЯМОЕ совпадение. Если вакансия требует Kubernetes, CI/CD, Terraform, а кандидат знает Python на уровне автоматизации — это НЕ совпадение.

4. ВЫСОКИЙ SCORE (70+) ставь ТОЛЬКО если:
   - Роль прямо совпадает с опытом кандидата (Product Manager, Innovation Lead, AI Product, Program Manager, Business Development)
   - Кандидат реально конкурентоспособен среди других претендентов
   - Нет критических пробелов в требованиях

5. СРЕДНИЙ SCORE (40-69): роль частично подходит, есть пересечения, но кандидат не идеальный кандидат.

6. НИЗКИЙ SCORE (0-39): роль не совпадает, или стек совсем другой, или грейд не тот.

7. Deal-breakers кандидата — если они нарушены, score = 0.

Верни JSON:
{
  "score": <число от 0 до 100>,
  "reason": "<краткое обоснование на русском, 2-3 предложения>",
  "strengths": ["сильная сторона 1", "сильная сторона 2"],
  "gaps": ["чего не хватает 1", "чего не хватает 2"]
}

Будь ЧЕСТНЫМ и СТРОГИМ. Лучше пропустить вакансию, чем предложить неподходящую.
Отвечай ТОЛЬКО валидным JSON."""


class VacancyMatcher:
    def __init__(self, llm: GroqClient, profile: dict, resume_text: str = ""):
        self.llm = llm
        self.profile = profile
        self.resume_text = resume_text

    @staticmethod
    def extract_resume_text(pdf_path: Path) -> str:
        """Extract text from PDF resume."""
        text_parts = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
        except Exception as e:
            logger.error("Failed to parse PDF {}: {}", pdf_path, e)
        return "\n".join(text_parts)

    def _build_candidate_context(self) -> str:
        """Build a text representation of the candidate profile."""
        p = self.profile
        parts = [
            f"Имя: {p.get('name', 'N/A')}",
            f"Текущая роль: {p.get('title', 'N/A')}",
            f"Опыт: {p.get('experience_years', 'N/A')} лет",
            f"Ключевой опыт: Product Management в Атом (2 года), Program Manager AI Business SPb, "
            f"Founder в ITMO TECH, Development Engineer в Airalab (4 года робототехника)",
            f"Навыки: {', '.join(p.get('skills', []))}",
            f"Языки: {', '.join(p.get('languages', []))}",
            f"Локация: {p.get('location', 'N/A')}",
            f"Удалёнка: {'Да' if p.get('remote_ok') else 'Нет'}",
            f"Желаемые роли: {', '.join(p.get('preferred_roles', []))}",
            f"Deal-breakers: {', '.join(p.get('deal_breakers', []))}",
            f"\nВАЖНО: кандидат НЕ чистый разработчик. Основной опыт — продуктовый менеджмент, "
            f"инновации, AI-продукты, кросс-функциональные команды. "
            f"Python используется для автоматизации и прототипов, а не для enterprise backend.",
        ]
        if p.get("salary_min"):
            parts.append(f"Мин. зарплата: {p['salary_min']}")
        if self.resume_text:
            parts.append(f"\n--- Резюме (PDF) ---\n{self.resume_text[:3000]}")
        return "\n".join(parts)

    async def match(self, vacancy: dict) -> dict:
        """
        Match a vacancy against the candidate profile.
        Returns dict with score, reason, strengths, gaps.
        """
        candidate_ctx = self._build_candidate_context()
        vacancy_text = (
            f"Вакансия: {vacancy.get('title', 'N/A')}\n"
            f"Компания: {vacancy.get('company', 'N/A')}\n"
            f"Локация: {vacancy.get('location', 'N/A')}\n"
            f"Зарплата: {vacancy.get('salary', 'N/A')}\n"
            f"Навыки: {vacancy.get('skills', 'N/A')}\n"
            f"Требования: {vacancy.get('requirements', 'N/A')}\n"
            f"Удалёнка: {vacancy.get('remote', 'N/A')}"
        )

        user_msg = (
            f"=== ПРОФИЛЬ КАНДИДАТА ===\n{candidate_ctx}\n\n"
            f"=== ВАКАНСИЯ ===\n{vacancy_text}"
        )

        try:
            result = await self.llm.chat_json(SYSTEM_PROMPT, user_msg)
            return result
        except Exception as e:
            logger.error("Matching failed: {}", e)
            return {"score": 0, "reason": f"Ошибка анализа: {e}", "strengths": [], "gaps": []}
