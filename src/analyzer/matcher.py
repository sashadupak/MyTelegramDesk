import pdfplumber
from pathlib import Path
from src.llm.groq_client import GroqClient
from loguru import logger

_IDENTITY_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "identity.md"
try:
    IDENTITY = _IDENTITY_PATH.read_text(encoding="utf-8")
except FileNotFoundError:
    IDENTITY = ""
    logger.warning("identity.md not found at {}", _IDENTITY_PATH)


SCORE_PROMPT = """Ты — личный карьерный advisor Александра. Ты знаешь его ГЛУБОКО.
Твоя задача — оценить вакансию как OPPORTUNITY, а не как "подхожу ли я по чеклисту".

{identity}

---

АЛГОРИТМ:

ШАГ 1. RED FLAGS — жёсткий стоп.
Если в вакансии есть хотя бы один red flag из профиля → score=0, не анализируй дальше.
Если роль из раздела "Где я НЕ выделюсь" → score=0.

ШАГ 2. SCORING по 6 критериям (каждый 0-5):

1. СВОБОДА ДЕЙСТВИЙ
   Сигналы: "launch", "build", "new direction", "0→1", "с нуля", "innovation"
   0 = жёсткие рамки, execution only | 5 = полная автономия

2. ДОСТУП К РЕСУРСАМ
   Корпорация/фонд/lab, бюджет, команды
   0 = "работа за идею" | 5 = серьёзный бюджет

3. БЛИЗОСТЬ К TOP MANAGEMENT
   Кому репортит? C-level / фаундеры?
   0 = 3+ уровня до CEO | 5 = напрямую фаундер

4. НЕОПРЕДЕЛЁННОСТЬ РОЛИ (плюс для Александра!)
   Размытое описание = можно "захватить" роль
   0 = жёсткий JD, checklist | 5 = "ищем человека, а не должность"

5. СООТВЕТСТВИЕ ДРАЙВЕРАМ
   AI, инновации, deep tech, новые продукты, ивенты, space
   0 = скучная индустрия | 5 = прямое попадание

6. ВОЗМОЖНОСТЬ СОЗДАТЬ НАПРАВЛЕНИЕ (самый важный ×2)
   Можно ли вырасти в юнит / продукт / бизнес?
   0 = потолок виден | 5 = роль может стать бизнесом

ШАГ 3. HIDDEN OPPORTUNITIES — ищи неявные сигналы:
- "начинаем новое направление", "формируем команду", "экспериментируем"
- Компания недавно привлекла инвестиции, делает R&D, запускает lab
- Роль настолько размыта что можно role-carve — предложить себя как решение
Эти сигналы дают +3-5 бонусных баллов.

ШАГ 4. TOP-5% CHECK:
Представь 100 кандидатов на эту роль.
Его суперсила: стык техники + бизнеса + стартап-опыт.
Его слабость: нет стабильного корп трека.
Он top-5 ТОЛЬКО если роль ценит нелинейный путь и предпринимательство выше стабильности.

ИТОГО:
score = (сумма 6 критериев) + (критерий 6 ещё раз, он ×2) + (hidden opportunity bonus)
Максимум = 35 + 5 = 40. Нормализуй в 0-100.

Верни JSON:
{{
  "score": <0-100>,
  "red_flags_found": ["flag1"] или [],
  "dimensions": {{
    "freedom": {{"score": <0-5>, "signal": "что увидел"}},
    "resources": {{"score": <0-5>, "signal": "..."}},
    "proximity_to_top": {{"score": <0-5>, "signal": "..."}},
    "role_ambiguity": {{"score": <0-5>, "signal": "..."}},
    "driver_fit": {{"score": <0-5>, "signal": "..."}},
    "new_direction_potential": {{"score": <0-5>, "signal": "..."}}
  }},
  "hidden_opportunities": ["сигнал 1"] или [],
  "hidden_bonus": <0-5>,
  "is_top5": true/false,
  "top5_reason": "почему top-5 или нет",
  "hook": "Одно предложение — что делает эту вакансию opportunity для Александра (или null)",
  "strategy": "role_carving | opportunity_driven | platform_entry | standard",
  "recommendation": "strong_apply | apply | skip",
  "reason": "2-3 предложения итоговой оценки"
}}

Будь ЖЁСТКИМ. Skip 95% вакансий. Показывай только те, где Александр — уникальное решение.
Отвечай ТОЛЬКО валидным JSON."""


class VacancyMatcher:
    def __init__(self, llm: GroqClient, profile: dict, resume_text: str = ""):
        self.llm = llm
        self.profile = profile
        self.resume_text = resume_text
        self.score_prompt = SCORE_PROMPT.format(identity=IDENTITY)

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

    async def match(self, vacancy: dict, raw_text: str = "") -> dict:
        """Single-pass deep analysis: red flags → 6D scoring → hidden opportunities → top-5%."""
        text = raw_text if raw_text else (
            f"Вакансия: {vacancy.get('title', 'N/A')}\n"
            f"Компания: {vacancy.get('company', 'N/A')}\n"
            f"Локация: {vacancy.get('location', 'N/A')}\n"
            f"Зарплата: {vacancy.get('salary', 'N/A')}\n"
            f"Навыки: {vacancy.get('skills', 'N/A')}\n"
            f"Требования: {vacancy.get('requirements', 'N/A')}\n"
            f"Удалёнка: {vacancy.get('remote', 'N/A')}"
        )

        try:
            result = await self.llm.chat_json(self.score_prompt, text)

            # Ensure required fields
            result.setdefault("score", 0)
            result.setdefault("recommendation", "skip")
            result.setdefault("reason", "")
            result.setdefault("hook", None)
            result.setdefault("is_top5", False)
            result.setdefault("red_flags_found", [])
            result.setdefault("hidden_opportunities", [])
            result.setdefault("dimensions", {})
            result.setdefault("strategy", "standard")

            # Override: if not top-5, force skip
            if not result["is_top5"] and result["score"] > 0:
                result["recommendation"] = "skip"
                result["reason"] += " [Не top-5 — пропускаем]"

            score = result["score"]
            rec = result["recommendation"]
            logger.info("'{}' — score:{} rec:{} strategy:{} top5:{}",
                        vacancy.get("title", "?"), score, rec,
                        result["strategy"], result["is_top5"])

            return result

        except Exception as e:
            logger.error("Matching failed: {}", e)
            return {
                "score": 0,
                "reason": f"Ошибка: {e}",
                "recommendation": "skip",
                "hook": None,
                "is_top5": False,
                "red_flags_found": [],
                "hidden_opportunities": [],
                "dimensions": {},
                "strategy": "standard",
            }
