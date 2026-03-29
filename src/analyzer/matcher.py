import pdfplumber
from pathlib import Path
from src.llm.groq_client import GroqClient
from loguru import logger

# Load identity profile
_IDENTITY_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "identity.md"
try:
    IDENTITY = _IDENTITY_PATH.read_text(encoding="utf-8")
except FileNotFoundError:
    IDENTITY = ""
    logger.warning("identity.md not found at {}", _IDENTITY_PATH)


HOOK_PROMPT = """Ты — личный карьерный advisor Александра. Ты знаешь его ГЛУБОКО — не просто резюме, а его ценности, цели, доменную экспертизу и что его реально зажигает.

ВОТ ЕГО ПОЛНЫЙ ПРОФИЛЬ:
{identity}

---

ЗАДАЧА: Прочитай вакансию и ответь на один вопрос:
"Есть ли здесь хотя бы ОДНА вещь, от которой Александр скажет — о, это про меня?"

Это может быть:
- Пересечение с его доменной экспертизой (AI agents, robotics, deep tech, space, innovation)
- Совпадение ценностей (масштаб, скорость, автономия, амбициозная команда)
- Роль на стыке техники и бизнеса — то, где его профиль уникален
- Компания/фаундер с track record, который резонирует
- Задача 0→1 (новый продукт, новый рынок, новое направление)

НЕ является hook-ом:
- Просто "хорошая зарплата" или "удалёнка" — это гигиенические факторы
- "Нужен Python разработчик" — он не чистый разработчик
- Размытые фразы типа "dynamic team" без конкретики
- Вакансия где его доменная экспертиза не даёт преимущества

Верни JSON:
{{
  "has_hook": true/false,
  "hook": "Что именно зацепило — конкретно (или null)",
  "domain_match": "Какая из его доменных экспертиз пересекается (или null)",
  "values_match": "Какие ценности совпадают (или null)",
  "hook_strength": <1-10>,
  "why_not": "Если нет hook-а — почему это не его (или null)"
}}

Будь ОЧЕНЬ избирателен. Hook должен быть только там, где есть РЕАЛЬНОЕ пересечение с его уникальным профилем.
Отвечай ТОЛЬКО валидным JSON."""


TOP5_PROMPT = """Ты — венчурный рекрутер с 10-летним опытом. Ты нанимал сотни людей и умеешь отличить реальный top-кандидата от "ну, подходит наверное".

ПРОФИЛЬ АЛЕКСАНДРА:
{identity}

---

ЗАДАЧА: На эту вакансию пришло 100 откликов. Александр — в top-5?

ПРАВИЛА:
1. Представь 100 реальных кандидатов. Кто они? Какой типичный профиль?
2. Где его УНИКАЛЬНОЕ сочетание (инженер + продукт + инновации + AI) даёт преимущество?
3. Где ему объективно НЕ ХВАТАЕТ по сравнению с другими?
4. Есть ли в вакансии требования из его раздела "Где я НЕ top-5%"?
5. Есть ли в вакансии пересечение с его разделом "Где я top-5%"?

Верни JSON:
{{
  "is_top5": true/false,
  "percentile": <1-100>,
  "typical_competitors": "2-3 типичных профиля конкурентов",
  "his_edge": "Конкретное преимущество (или null)",
  "his_weakness": "Конкретная слабость (или null)",
  "verdict": "Одно предложение — почему top-5 или нет"
}}

Top-5 из 100 — это серьёзно. Давай только там, где его профиль РЕАЛЬНО уникален для этой роли.
Отвечай ТОЛЬКО валидным JSON."""


class VacancyMatcher:
    def __init__(self, llm: GroqClient, profile: dict, resume_text: str = ""):
        self.llm = llm
        self.profile = profile
        self.resume_text = resume_text
        self.hook_prompt = HOOK_PROMPT.format(identity=IDENTITY)
        self.top5_prompt = TOP5_PROMPT.format(identity=IDENTITY)

    @staticmethod
    def extract_resume_text(pdf_path: Path) -> str:
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

    async def find_hook(self, vacancy: dict, raw_text: str = "") -> dict:
        text = raw_text if raw_text else (
            f"{vacancy.get('title', '')}\n{vacancy.get('company', '')}\n"
            f"{vacancy.get('requirements', '')}\n{vacancy.get('skills', '')}"
        )
        try:
            return await self.llm.chat_json(self.hook_prompt, text)
        except Exception as e:
            logger.error("Hook detection failed: {}", e)
            return {"has_hook": False, "hook_strength": 0}

    async def check_top5(self, vacancy: dict, raw_text: str = "") -> dict:
        vacancy_text = raw_text if raw_text else (
            f"Вакансия: {vacancy.get('title', 'N/A')}\n"
            f"Компания: {vacancy.get('company', 'N/A')}\n"
            f"Локация: {vacancy.get('location', 'N/A')}\n"
            f"Зарплата: {vacancy.get('salary', 'N/A')}\n"
            f"Навыки: {vacancy.get('skills', 'N/A')}\n"
            f"Требования: {vacancy.get('requirements', 'N/A')}\n"
            f"Удалёнка: {vacancy.get('remote', 'N/A')}"
        )
        try:
            return await self.llm.chat_json(self.top5_prompt, vacancy_text)
        except Exception as e:
            logger.error("Top-5 check failed: {}", e)
            return {"is_top5": False, "percentile": 100}

    async def match(self, vacancy: dict, raw_text: str = "") -> dict:
        # Stage 1: Hook — is there something that resonates with my identity?
        hook_result = await self.find_hook(vacancy, raw_text)
        has_hook = hook_result.get("has_hook", False)
        hook_strength = hook_result.get("hook_strength", 0)

        if not has_hook or hook_strength < 6:
            reason_parts = []
            if hook_result.get("why_not"):
                reason_parts.append(hook_result["why_not"])
            if not hook_result.get("domain_match"):
                reason_parts.append("Нет пересечения по доменной экспертизе")
            if not hook_result.get("values_match"):
                reason_parts.append("Ценности не совпадают")

            logger.info("No hook for '{}': {}",
                        vacancy.get('title', '?'),
                        '; '.join(reason_parts) or 'no match')
            return {
                "score": 0,
                "reason": '; '.join(reason_parts) or "Ничего не зацепило",
                "strengths": [],
                "gaps": [],
                "recommendation": "skip",
                "hook": hook_result,
                "top5": None,
            }

        # Stage 2: Top-5% — am I a killer candidate?
        top5_result = await self.check_top5(vacancy, raw_text)
        is_top5 = top5_result.get("is_top5", False)
        percentile = top5_result.get("percentile", 100)

        if not is_top5:
            score = max(10, 100 - percentile)
            logger.info("Hook but not top-5 for '{}': {} (percentile {})",
                        vacancy.get('title', '?'), top5_result.get('verdict', ''), percentile)
            return {
                "score": score,
                "reason": f"Hook: {hook_result.get('hook', '?')}. "
                          f"Но не top-5: {top5_result.get('verdict', '?')}",
                "strengths": [hook_result.get("hook", "")],
                "gaps": [top5_result.get("his_weakness", "")],
                "recommendation": "skip",
                "hook": hook_result,
                "top5": top5_result,
            }

        # Both passed
        score = min(100, 60 + hook_strength * 2 + (100 - percentile))
        return {
            "score": score,
            "reason": f"🎯 {hook_result.get('hook', '?')}. "
                      f"Домен: {hook_result.get('domain_match', '?')}. "
                      f"Top-{percentile}: {top5_result.get('verdict', '?')}",
            "strengths": [
                hook_result.get("hook", ""),
                hook_result.get("domain_match", ""),
                top5_result.get("his_edge", ""),
            ],
            "gaps": [top5_result.get("his_weakness", "")] if top5_result.get("his_weakness") else [],
            "recommendation": "strong_apply" if hook_strength >= 8 and percentile <= 3 else "apply",
            "hook": hook_result,
            "top5": top5_result,
        }
