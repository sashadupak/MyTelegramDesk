import pdfplumber
from pathlib import Path
from src.llm.groq_client import GroqClient
from loguru import logger

# Stage 1: Is there a HOOK? One thing that makes this vacancy special for this candidate?
HOOK_PROMPT = """Ты — личный карьерный советник Александра Дупака. Ты знаешь его глубоко.

ЕГО ПРОФИЛЬ:
- Траектория: Engineer (робототехника 10 лет) → Founder (EdTech, AI-сервисы) → Product Manager (Атом, 10+ команд) → Innovation & AI Lead
- Суперсила: находить новые бизнес-возможности там, где другие видят путь из А в Б
- Уникальность: редкое сочетание глубокого инженерного бэкграунда + продуктового мышления + венчурной среды
- Драйвит: влияние на масштабе, риск, рост ×10, новые рынки
- Среда: стартапы, венчур, корпоративные инновации, AI, deep tech, space tech
- Сейчас: AI Business SPb, хакатоны, партнёрства, Claude Agents, ETH Zürich New Space Economy
- НЕ хочет: рутину, линейные позиции, аутсорс, legacy, "работу за идею"

ЗАДАЧА: Прочитай вакансию и найди ОДНУ вещь — hook — которая бы заставила Александра остановиться и сказать "О, это интересно!".

Hook может быть:
- Компания делает что-то на стыке его интересов (AI + space, robotics + бизнес, deep tech + product)
- Роль подразумевает создание чего-то нового с нуля (new product, new market, 0→1)
- Команда/фаундер с сильным трекшном и амбицией
- Уникальная возможность применить его сочетание инженер + продукт + инновации
- Среда: YC, топ-фонды, быстрый рост, серия A+, корпоративный venture
- Прямое пересечение с его текущим фокусом (AI agents, new space, innovation programs)

НЕ является hook-ом:
- "Нужен Python разработчик" (он не чистый разработчик)
- "Хорошая зарплата" (это не мотиватор)
- "Удалёнка" (это базовое требование, не hook)
- "Большая компания" без инновационного контекста

Верни JSON:
{
  "has_hook": true/false,
  "hook": "Одно предложение — что именно цепляет (или null)",
  "hook_strength": <1-10, где 10 = 'бросить всё и откликнуться'>,
  "why_not": "Если нет hook-а — почему эта вакансия скучная/не та (или null)"
}

Будь ОЧЕНЬ избирателен. Hook должен быть у максимум 10-15% вакансий.
Отвечай ТОЛЬКО валидным JSON."""


# Stage 2: Am I top 5%?
TOP_CANDIDATE_PROMPT = """Ты — венчурный рекрутер, который нанимал сотни людей. Ты умеешь оценить, кто реально top-кандидат, а кто "ну подходит, наверное".

ЗАДАЧА: Представь, что на эту вакансию пришло 100 откликов. Оцени — Александр Дупак попадёт в топ-5 кандидатов?

ЕГО ПРОФИЛЬ:
- 10 лет робототехника: дроны, манипуляторы, CV, нелинейные контроллеры, Airalab (4 года)
- Founder: EdTech платформа (топ-3 BrainBox), AI-агрегатор (1000 юзеров, грант ФСИ)
- Product Manager Атом: 10+ команд, подрядчики РФ/Китай/Корея, delivery с нуля, AI в продуктовый цикл
- Program Manager AI Business SPb: хакатоны, партнёрства, AI Shopping Assistant (доклад ИТМО)
- ИТМО: бакалавр CS/Robotics + магистр Innovation Entrepreneurship (English)
- Сертификации: AltaLab (Altair Capital), AI Talent Hub (Claude Agents), ETH Zürich (Space), PMI-ACP
- Английский C1
- Выступления: Skolkovo, SPB Founders, Product Camp, 50+ конференций
- Навыки: Product Management, AI/LLM, Python, Cross-functional Leadership, BizDev, Innovation

ПРАВИЛА ОЦЕНКИ:

1. Представь 100 типичных кандидатов на эту роль. Кто они? Какой у них опыт?
2. Чем Александр ОБЪЕКТИВНО сильнее большинства из них?
3. Чем он ОБЪЕКТИВНО слабее? Что есть у других, чего нет у него?
4. Попадёт ли он в top-5 из 100?

КРАСНЫЕ ФЛАГИ (автоматически = НЕ top-5):
- Вакансия требует 5+ лет в узком стеке (ML research, backend Java, DevOps), которого у него нет
- Роль чисто техническая без продуктовой/бизнес-составляющей
- Нужен отраслевой опыт, которого нет (финтех compliance, медицина, юридический)
- Уровень C-level в enterprise (CEO, CTO крупной компании) — рано

ЗЕЛЁНЫЕ ФЛАГИ (усиливают позицию):
- Нужен человек на стыке техники и бизнеса — он один из немногих
- Стартап ищет product + technical background — идеально
- AI/innovation роль, где нужен практический опыт запуска — его конёк
- Нужен человек с предпринимательским мышлением в корпорации — точное попадание

Верни JSON:
{
  "is_top5": true/false,
  "percentile": <1-100, где 1 = лучший кандидат из 100>,
  "typical_competitors": "Кто обычно претендует на эту роль (2-3 примера профилей)",
  "his_edge": "В чём он сильнее типичных кандидатов (или null)",
  "his_weakness": "В чём слабее (или null)",
  "verdict": "Одно предложение: почему top-5 или почему нет"
}

Будь ЖЁСТКИМ и ЧЕСТНЫМ. Top-5 из 100 — это действительно сильная позиция. Не раздавай её щедро.
Отвечай ТОЛЬКО валидным JSON."""


class VacancyMatcher:
    def __init__(self, llm: GroqClient, profile: dict, resume_text: str = ""):
        self.llm = llm
        self.profile = profile
        self.resume_text = resume_text

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
        """Stage 1: Is there something that hooks the candidate?"""
        text = raw_text if raw_text else (
            f"{vacancy.get('title', '')}\n{vacancy.get('company', '')}\n"
            f"{vacancy.get('requirements', '')}\n{vacancy.get('skills', '')}"
        )
        try:
            return await self.llm.chat_json(HOOK_PROMPT, text)
        except Exception as e:
            logger.error("Hook detection failed: {}", e)
            return {"has_hook": False, "hook_strength": 0}

    async def check_top5(self, vacancy: dict) -> dict:
        """Stage 2: Am I a top-5% candidate for this role?"""
        vacancy_text = (
            f"Вакансия: {vacancy.get('title', 'N/A')}\n"
            f"Компания: {vacancy.get('company', 'N/A')}\n"
            f"Локация: {vacancy.get('location', 'N/A')}\n"
            f"Зарплата: {vacancy.get('salary', 'N/A')}\n"
            f"Навыки: {vacancy.get('skills', 'N/A')}\n"
            f"Требования: {vacancy.get('requirements', 'N/A')}\n"
            f"Удалёнка: {vacancy.get('remote', 'N/A')}"
        )
        try:
            return await self.llm.chat_json(TOP_CANDIDATE_PROMPT, vacancy_text)
        except Exception as e:
            logger.error("Top-5 check failed: {}", e)
            return {"is_top5": False, "percentile": 100}

    async def match(self, vacancy: dict, raw_text: str = "") -> dict:
        """
        Two-stage matching:
        1. Hook — is there ONE thing that makes me stop and say 'wow'?
        2. Top-5% — am I a killer candidate, not just 'one of many'?
        """
        # Stage 1: Hook
        hook_result = await self.find_hook(vacancy, raw_text)
        has_hook = hook_result.get("has_hook", False)
        hook_strength = hook_result.get("hook_strength", 0)

        if not has_hook or hook_strength < 5:
            logger.info("No hook for '{}': {}",
                        vacancy.get('title', '?'),
                        hook_result.get('why_not', 'no reason'))
            return {
                "score": 0,
                "reason": hook_result.get("why_not", "Ничего не зацепило"),
                "strengths": [],
                "gaps": [],
                "recommendation": "skip",
                "hook": hook_result,
                "top5": None,
            }

        # Stage 2: Top-5% check
        top5_result = await self.check_top5(vacancy)
        is_top5 = top5_result.get("is_top5", False)
        percentile = top5_result.get("percentile", 100)

        if not is_top5:
            score = max(10, 100 - percentile)  # low score for non-top5
            logger.info("Hook found but not top-5 for '{}': percentile {}",
                        vacancy.get('title', '?'), percentile)
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

        # Both stages passed — this is a real match
        score = min(100, 60 + hook_strength * 2 + (100 - percentile))
        return {
            "score": score,
            "reason": f"🎯 Hook: {hook_result.get('hook', '?')}. "
                      f"Top-{percentile} из 100: {top5_result.get('verdict', '?')}",
            "strengths": [hook_result.get("hook", ""), top5_result.get("his_edge", "")],
            "gaps": [top5_result.get("his_weakness", "")] if top5_result.get("his_weakness") else [],
            "recommendation": "strong_apply" if hook_strength >= 8 and percentile <= 3 else "apply",
            "hook": hook_result,
            "top5": top5_result,
        }
