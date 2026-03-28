import aiohttp
from bs4 import BeautifulSoup
from src.llm.groq_client import GroqClient
from loguru import logger

SYSTEM_PROMPT = """Ты — аналитик, составляющий краткое досье о компании для соискателя.
На основе предоставленной информации составь краткое досье на русском языке.

Верни JSON:
{
  "summary": "Краткое описание компании (2-3 предложения)",
  "industry": "Отрасль",
  "size": "Размер компании если известно",
  "culture_hints": "Намёки на культуру компании",
  "recent_news": "Интересные факты или недавние новости",
  "talking_points": ["зацепка для сопроводительного 1", "зацепка 2", "зацепка 3"]
}

talking_points — это конкретные факты о компании, которые кандидат может упомянуть
в сопроводительном письме, чтобы показать свою осведомлённость и заинтересованность.

Отвечай ТОЛЬКО валидным JSON."""


class CompanyResearcher:
    def __init__(self, llm: GroqClient):
        self.llm = llm

    async def _search_web(self, query: str) -> str:
        """Simple web search via DuckDuckGo HTML."""
        url = "https://html.duckduckgo.com/html/"
        headers = {"User-Agent": "Mozilla/5.0 (compatible; JobSeeker/1.0)"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data={"q": query},
                                        headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status != 200:
                        return ""
                    html = await resp.text()
                    soup = BeautifulSoup(html, "html.parser")
                    results = []
                    for r in soup.select(".result__body")[:5]:
                        title_el = r.select_one(".result__title")
                        snippet_el = r.select_one(".result__snippet")
                        title = title_el.get_text(strip=True) if title_el else ""
                        snippet = snippet_el.get_text(strip=True) if snippet_el else ""
                        if title or snippet:
                            results.append(f"{title}: {snippet}")
                    return "\n".join(results)
        except Exception as e:
            logger.error("Web search failed for '{}': {}", query, e)
            return ""

    async def research(self, company_name: str, vacancy_title: str = "") -> dict:
        """
        Research a company and return structured info.
        """
        if not company_name or company_name.lower() == "null":
            return {
                "summary": "Компания не указана в вакансии",
                "industry": "Неизвестно",
                "size": "Неизвестно",
                "culture_hints": "Нет данных",
                "recent_news": "Нет данных",
                "talking_points": [],
            }

        search_results = await self._search_web(
            f"{company_name} компания отзывы сотрудников"
        )
        search_results_2 = await self._search_web(
            f"{company_name} {vacancy_title} новости"
        )

        context = (
            f"Компания: {company_name}\n"
            f"Вакансия: {vacancy_title}\n\n"
            f"--- Результаты поиска 1 ---\n{search_results}\n\n"
            f"--- Результаты поиска 2 ---\n{search_results_2}"
        )

        try:
            result = await self.llm.chat_json(SYSTEM_PROMPT, context)
            return result
        except Exception as e:
            logger.error("Company research failed for '{}': {}", company_name, e)
            return {
                "summary": f"Не удалось исследовать компанию: {e}",
                "industry": "Неизвестно",
                "size": "Неизвестно",
                "culture_hints": "Нет данных",
                "recent_news": "Нет данных",
                "talking_points": [],
            }
