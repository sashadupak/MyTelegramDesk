import pdfplumber
from pathlib import Path
from src.llm.groq_client import GroqClient
from loguru import logger

# Stage 1: Is this a "venture job" at all?
VENTURE_FILTER_PROMPT = """Ты — эксперт по венчурному рынку труда. Определи, является ли вакансия "venture job".

Venture Job — это вакансия в среде риска, скорости и роста ×10. Это НЕ про название позиции, а про СРЕДУ и АМБИЦИЮ команды.

ДА — venture job, если:
- Стартап (особенно с инвестициями, акселераторами, трекшном)
- Венчурный фонд, акселератор, инвестиционная платформа
- Корпоративные инновации (новый продукт, новое направление, R&D lab)
- Быстрорастущая tech-компания на стадии scale-up
- AI/ML/DeepTech/SpaceTech/FinTech/EdTech команда с амбициозной миссией
- Роли: product, bizdev, investment analyst, scouting, deal flow, new business, growth, operations в growth-среде, engineering lead в стартапе

НЕТ — не venture job, если:
- Линейная операционная позиция в стабильной компании
- Классический малый бизнес (кафе, салоны, локальные услуги)
- Агентство/аутсорс (кроме работающих со стартапами)
- "Работа за идею" без нормальной компенсации
- Чистая разработка в legacy-проекте без инновационного контекста
- Стажировки ради опыта
- Линейный backend/frontend/QA в аутсорсе или банке

Верни JSON:
{
  "is_venture": true/false,
  "venture_signals": ["сигнал 1", "сигнал 2"],
  "red_flags": ["красный флаг 1"],
  "environment_type": "startup | scaleup | corporate_innovation | vc_fund | agency | traditional_corp | small_business | other",
  "confidence": <0.0-1.0>
}

Отвечай ТОЛЬКО валидным JSON."""

# Stage 2: Deep candidate-vacancy fit
MATCH_PROMPT = """Ты — венчурный рекрутер с 10-летним опытом. Оцени РЕАЛИСТИЧНО, насколько кандидат конкурентоспособен на эту вакансию. Ты видел тысячи кандидатов и умеешь отличать реальный fit от натяжки.

АЛГОРИТМ ОЦЕНКИ (5 измерений, каждое 0-20 баллов):

1. ROLE FIT (0-20): Совпадает ли роль с траекторией кандидата?
   - 16-20: Прямое попадание (Product Manager → Product Manager, Innovation Lead → Innovation Lead)
   - 10-15: Смежная роль (Product Manager → BizDev, Program Manager → Operations Lead)
   - 5-9: Частичное пересечение (инженер-робототехник → ML Engineer, но без ML опыта)
   - 0-4: Роль из другой области (Product Manager → Senior Java Backend Developer)

2. SKILLS MATCH (0-20): Реальное совпадение hard skills
   - Считай только ПРЯМОЙ опыт, не "может разобраться"
   - Python для автоматизации ≠ Python для enterprise backend
   - Product Management, Cross-functional leadership, AI/LLM, Innovation — это его сильные стороны
   - Kubernetes, Terraform, System Design — НЕ его стек

3. SENIORITY FIT (0-20): Соответствует ли грейд?
   - Кандидат: 7 лет общего опыта, но специфика в product + robotics + innovation
   - Если вакансия требует "5+ лет чистого backend" — это НЕ его уровень в этом стеке
   - Если вакансия на Junior/Intern — overqualified
   - Если вакансия на C-level enterprise — underqualified

4. COMPETITIVE EDGE (0-20): Насколько кандидат сильнее среднего претендента?
   - Уникальное сочетание: инженерный бэкграунд + продукт + инновации + AI
   - ИТМО, Атом, AI Business SPb, AltaLab — это конкурентные преимущества
   - Но если вакансия требует то, чего у него нет (5 лет ML research, MBA) — это слабость
   - Представь 10 типичных кандидатов на эту роль. Будет ли этот в top-3?

5. GROWTH POTENTIAL (0-20): Потенциал взаимного роста
   - Может ли кандидат вырасти ×10 в этой роли?
   - Может ли компания получить уникальную ценность от его профиля?
   - Есть ли синергия между его суперсилой (находить новые возможности) и задачами роли?

ИТОГО: score = сумма пяти измерений (0-100)

Верни JSON:
{
  "score": <0-100>,
  "dimensions": {
    "role_fit": {"score": <0-20>, "comment": "..."},
    "skills_match": {"score": <0-20>, "comment": "..."},
    "seniority_fit": {"score": <0-20>, "comment": "..."},
    "competitive_edge": {"score": <0-20>, "comment": "..."},
    "growth_potential": {"score": <0-20>, "comment": "..."}
  },
  "reason": "<2-3 предложения итоговой оценки на русском>",
  "strengths": ["сильная сторона 1", "сильная сторона 2"],
  "gaps": ["чего не хватает 1", "чего не хватает 2"],
  "recommendation": "strong_apply | apply | maybe | skip"
}

Будь ЧЕСТНЫМ. Лучше пропустить 10 вакансий, чем предложить 1 неподходящую.
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
        """Build a detailed text representation of the candidate."""
        p = self.profile
        return f"""Имя: {p.get('name', 'N/A')}
Текущая роль: {p.get('title', 'N/A')}
Общий опыт: {p.get('experience_years', 'N/A')} лет

КАРЬЕРНАЯ ТРАЕКТОРИЯ (Engineer → Founder → Product):

1. Робототехника (10 лет, с 10 лет): 33 призовых места, собственная робо-платформа,
   дроны, манипуляторы, компьютерное зрение, нелинейные контроллеры.
   Airalab — Development Engineer (4 года): проекты для нефтепереработки, беспилотные авто.

2. Предпринимательство: Founder в ITMO TECH — EdTech платформа (топ-3 BrainBox),
   AI-агрегатор мероприятий (1000 пользователей, партнёры Napoleon IT), грант ФСИ.
   Выступления: SPB Founders, Skolkovo Startup Village, Product Camp.

3. Product Management: Атом (2 года) — координация 10+ продуктовых и инженерных команд,
   подрядчики из России, Китая, Южной Кореи. Delivery management с нуля.
   Инициировал внедрение AI в продуктовый цикл.

4. Сейчас: Program Manager AI Business SPb — хакатоны, партнёрства, AI-проекты для бизнеса.
   AI Shopping Assistant (доклад на Конгрессе молодых учёных ИТМО, статья в журнале).

ОБРАЗОВАНИЕ: ИТМО — бакалавр CS/Robotics + магистр Innovation Entrepreneurship (English).
Сертификации: AltaLab (Altair Capital), AI Product Engineering (AI Talent Hub),
New Space Economy (ETH Zürich), PMI-ACP, МФТИ (лидерство акселераторов).

Навыки: {', '.join(p.get('skills', []))}
Языки: {', '.join(p.get('languages', []))}
Локация: {p.get('location', 'N/A')} | Удалёнка: {'Да' if p.get('remote_ok') else 'Нет'}

Желаемые роли: {', '.join(p.get('preferred_roles', []))}
Deal-breakers: {', '.join(p.get('deal_breakers', []))}

СУПЕРСИЛА: находить новые бизнес-возможности там, где другие видят лишь путь из А в Б.
Уникальное сочетание: глубокий инженерный бэкграунд + продуктовое мышление + венчурная среда."""

    async def is_venture_job(self, vacancy: dict, raw_text: str = "") -> dict:
        """Stage 1: Filter — is this a venture job?"""
        text = raw_text or (
            f"Вакансия: {vacancy.get('title', 'N/A')}\n"
            f"Компания: {vacancy.get('company', 'N/A')}\n"
            f"Описание: {vacancy.get('requirements', 'N/A')}\n"
            f"Навыки: {vacancy.get('skills', 'N/A')}"
        )
        try:
            return await self.llm.chat_json(VENTURE_FILTER_PROMPT, text)
        except Exception as e:
            logger.error("Venture filter failed: {}", e)
            return {"is_venture": False, "confidence": 0}

    async def match(self, vacancy: dict, raw_text: str = "") -> dict:
        """
        Two-stage matching:
        1. Venture filter — skip non-venture jobs
        2. Deep multi-dimensional scoring
        """
        # Stage 1: Venture filter
        venture_result = await self.is_venture_job(vacancy, raw_text)

        if not venture_result.get("is_venture", False):
            logger.info("Skipped non-venture job: '{}' ({})",
                        vacancy.get('title', '?'),
                        venture_result.get('environment_type', '?'))
            return {
                "score": 0,
                "reason": f"Не venture job: {venture_result.get('environment_type', 'unknown')}. "
                          f"Red flags: {', '.join(venture_result.get('red_flags', []))}",
                "strengths": [],
                "gaps": [],
                "recommendation": "skip",
                "venture_filter": venture_result,
            }

        # Stage 2: Deep matching
        candidate_ctx = self._build_candidate_context()
        vacancy_text = (
            f"Вакансия: {vacancy.get('title', 'N/A')}\n"
            f"Компания: {vacancy.get('company', 'N/A')}\n"
            f"Локация: {vacancy.get('location', 'N/A')}\n"
            f"Зарплата: {vacancy.get('salary', 'N/A')}\n"
            f"Навыки: {vacancy.get('skills', 'N/A')}\n"
            f"Требования: {vacancy.get('requirements', 'N/A')}\n"
            f"Удалёнка: {vacancy.get('remote', 'N/A')}\n"
            f"Контекст среды: {venture_result.get('environment_type', 'N/A')}\n"
            f"Venture-сигналы: {', '.join(venture_result.get('venture_signals', []))}"
        )

        user_msg = (
            f"=== ПРОФИЛЬ КАНДИДАТА ===\n{candidate_ctx}\n\n"
            f"=== ВАКАНСИЯ ===\n{vacancy_text}"
        )

        try:
            result = await self.llm.chat_json(MATCH_PROMPT, user_msg)
            result["venture_filter"] = venture_result
            return result
        except Exception as e:
            logger.error("Matching failed: {}", e)
            return {
                "score": 0,
                "reason": f"Ошибка анализа: {e}",
                "strengths": [],
                "gaps": [],
                "recommendation": "skip",
            }
