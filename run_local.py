#!/usr/bin/env python3
"""
Local worker — runs on your laptop.
Handles: Groq LLM analysis (extract, match, research, cover letter).
Connects to server API via SSH tunnel.
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import aiohttp
from loguru import logger

from config.settings import settings
from src.llm.groq_client import GroqClient
from src.analyzer.extractor import VacancyExtractor
from src.analyzer.matcher import VacancyMatcher
from src.analyzer.researcher import CompanyResearcher
from src.generator.cover_letter import CoverLetterGenerator

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | <cyan>{name}</cyan> — <level>{message}</level>",
    level="INFO",
)

# Server API via SSH tunnel
SERVER_API = "http://127.0.0.1:8642"


class LocalWorker:
    def __init__(self):
        self.llm = GroqClient(
            api_key=settings.llm_api_key or settings.groq_api_key,
            model=settings.llm_model,
            provider=settings.llm_provider,
            fallback_api_key=settings.groq_api_key if settings.llm_provider != "groq" else None,
            fallback_provider="groq" if settings.llm_provider != "groq" else None,
            fallback_model=settings.groq_model,
        )
        self.extractor = VacancyExtractor(self.llm)

        self.profile = settings.load_profile()
        resume_text = ""
        if settings.resume_path.exists():
            resume_text = VacancyMatcher.extract_resume_text(settings.resume_path)
            logger.info("Loaded resume ({} chars)", len(resume_text))

        self.matcher = VacancyMatcher(self.llm, self.profile, resume_text)
        self.researcher = CompanyResearcher(self.llm)
        self.cover_gen = CoverLetterGenerator(self.llm, self.profile)

    async def process_vacancy(self, session: aiohttp.ClientSession, vacancy: dict) -> None:
        """Process a single vacancy through Groq pipeline."""
        vacancy_id = vacancy["id"]
        raw_text = vacancy["raw_text"]

        logger.info("Processing vacancy #{} ...", vacancy_id)

        channel = vacancy.get("channel_username", "")

        # 1. Extract structured data
        vacancy_data = await self.extractor.extract(raw_text)
        if not vacancy_data:
            logger.debug("#{}: not a vacancy (channel: @{})", vacancy_id, channel)
            await session.post(
                f"{SERVER_API}/api/vacancies/{vacancy_id}/result",
                json={
                    "match_result": {"score": 0, "reason": "Not a vacancy"},
                    "is_vacancy": False,
                    "channel_username": channel,
                },
            )
            return

        # 2. Score as opportunity (6D + hidden + top-5%)
        match_result = await self.matcher.match(vacancy_data, raw_text=raw_text)
        score = match_result.get("score", 0)
        rec = match_result.get("recommendation", "skip")
        strategy = match_result.get("strategy", "")
        logger.info("#{} '{}' — score:{} rec:{} strategy:{}",
                     vacancy_id, vacancy_data.get("title", "?"), score, rec, strategy)

        if rec == "skip" or score < settings.match_threshold:
            await session.post(
                f"{SERVER_API}/api/vacancies/{vacancy_id}/result",
                json={
                    "vacancy_data": vacancy_data,
                    "match_result": match_result,
                    "is_vacancy": True,
                    "channel_username": channel,
                },
            )
            return

        # 3. Research company
        company_info = await self.researcher.research(
            vacancy_data.get("company", ""),
            vacancy_data.get("title", ""),
        )

        # 4. Generate cover letter
        cover_result = await self.cover_gen.generate(vacancy_data, company_info, match_result)
        cover_letter = cover_result.get("cover_letter", "")

        # 5. Send everything back to server
        await session.post(
            f"{SERVER_API}/api/vacancies/{vacancy_id}/result",
            json={
                "vacancy_data": vacancy_data,
                "match_result": match_result,
                "company_info": company_info,
                "cover_letter": cover_letter,
            },
        )
        logger.info("Vacancy #{} — sent to server (score: {}, cover letter generated)", vacancy_id, score)

    async def poll_loop(self):
        """Poll server for unprocessed vacancies."""
        logger.info("Local worker started. Polling {} ...", SERVER_API)

        # Check server health
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{SERVER_API}/api/health") as resp:
                    health = await resp.json()
                    logger.info("Server OK: {}", health)
            except aiohttp.ClientError as e:
                logger.error("Cannot reach server at {}. Is SSH tunnel running?\n"
                             "  ssh -L 8642:127.0.0.1:8642 -i ~/.ssh/timeweb_key root@81.200.145.9",
                             SERVER_API)
                return

        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    # Fetch unprocessed
                    async with session.get(f"{SERVER_API}/api/vacancies/unprocessed") as resp:
                        vacancies = await resp.json()

                    if vacancies:
                        logger.info("Found {} unprocessed vacancies", len(vacancies))
                        for v in vacancies:
                            await self.process_vacancy(session, v)
                            await asyncio.sleep(1)  # Groq rate limit
                    else:
                        logger.debug("No new vacancies")

            except aiohttp.ClientError as e:
                logger.warning("Server connection error: {}. Retrying in 30s...", e)

            await asyncio.sleep(30)  # Poll every 30 seconds


if __name__ == "__main__":
    try:
        asyncio.run(LocalWorker().poll_loop())
    except KeyboardInterrupt:
        logger.info("Local worker stopped.")
