import asyncio
import json
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from loguru import logger

from config.settings import settings
from src.database.models import Database, init_db
from src.llm.groq_client import GroqClient
from src.parser.telegram_parser import TelegramParser
from src.analyzer.extractor import VacancyExtractor
from src.analyzer.matcher import VacancyMatcher
from src.analyzer.researcher import CompanyResearcher
from src.generator.cover_letter import CoverLetterGenerator
from src.bot.notification_bot import NotificationBot


class JobSeekerAgent:
    def __init__(self):
        self.db = Database(settings.db_path)
        self.llm = GroqClient(
            api_key=settings.llm_api_key or settings.groq_api_key,
            model=settings.llm_model,
            provider=settings.llm_provider,
            fallback_api_key=settings.groq_api_key if settings.llm_provider != "groq" else None,
            fallback_provider="groq" if settings.llm_provider != "groq" else None,
            fallback_model=settings.groq_model,
        )
        self.parser = TelegramParser(
            api_id=settings.telegram_api_id,
            api_hash=settings.telegram_api_hash,
            phone=settings.telegram_phone,
            session_path=settings.session_path,
        )
        self.extractor = VacancyExtractor(self.llm)

        # Load profile and resume
        self.profile = settings.load_profile()
        resume_text = ""
        if settings.resume_path.exists():
            resume_text = VacancyMatcher.extract_resume_text(settings.resume_path)
            logger.info("Loaded resume ({} chars)", len(resume_text))

        self.matcher = VacancyMatcher(self.llm, self.profile, resume_text)
        self.researcher = CompanyResearcher(self.llm)
        self.cover_gen = CoverLetterGenerator(self.llm, self.profile)
        self.bot = NotificationBot(
            token=settings.telegram_bot_token,
            owner_id=settings.telegram_owner_id,
            db=self.db,
        )
        self.scheduler = AsyncIOScheduler()

    async def init(self) -> None:
        """Initialize all components."""
        # Ensure data directory exists
        settings.data_dir.mkdir(parents=True, exist_ok=True)

        # Init database
        await init_db(settings.db_path)

        # Sync channels from config
        channels = settings.load_channels()
        for ch in channels:
            await self.db.upsert_channel(
                username=ch["username"],
                name=ch.get("name"),
                category=ch.get("category"),
                enabled=ch.get("enabled", True),
            )
        logger.info("Synced {} channels from config", len(channels))

        # Start Telethon
        await self.parser.start()

        # Build bot
        self.bot.build()

    async def process_channel(self, channel: dict) -> int:
        """
        Process a single channel: fetch new messages, extract vacancies,
        match, research, generate cover letters, notify.
        Returns number of matched vacancies.
        """
        username = channel["username"]
        min_id = channel.get("last_message_id", 0) or 0
        matched_count = 0

        messages = await self.parser.get_new_messages(username, min_id=min_id)
        if not messages:
            return 0

        max_msg_id = max(m["message_id"] for m in messages)

        for msg in messages:
            # 1. Extract vacancy structure
            vacancy_data = await self.extractor.extract(msg["text"])
            if not vacancy_data:
                continue

            # 2. Save to DB
            vacancy_id = await self.db.insert_vacancy(
                channel_username=username,
                message_id=msg["message_id"],
                raw_text=msg["text"],
                title=vacancy_data.get("title"),
                company=vacancy_data.get("company"),
                location=vacancy_data.get("location"),
                salary=vacancy_data.get("salary"),
                skills=vacancy_data.get("skills"),
                requirements=vacancy_data.get("requirements"),
                remote=vacancy_data.get("remote"),
                contact=vacancy_data.get("contact"),
                source_url=vacancy_data.get("source_url") or msg["url"],
            )
            if vacancy_id is None:
                continue  # duplicate

            # 3. Match against profile
            match_result = await self.matcher.match(vacancy_data)
            score = match_result.get("score", 0)
            await self.db.update_vacancy_match(
                vacancy_id, score, match_result.get("reason", "")
            )

            if score < settings.match_threshold:
                logger.debug("Vacancy #{} score {} < threshold {}, skipping",
                             vacancy_id, score, settings.match_threshold)
                continue

            # 4. Research company
            company_info = await self.researcher.research(
                vacancy_data.get("company", ""),
                vacancy_data.get("title", ""),
            )

            # 5. Generate cover letter
            cover_result = await self.cover_gen.generate(
                vacancy_data, company_info, match_result
            )
            cover_letter = cover_result.get("cover_letter", "")

            # 6. Save application
            app_id = await self.db.create_application(
                vacancy_id=vacancy_id,
                cover_letter=cover_letter,
                company_info=json.dumps(company_info, ensure_ascii=False),
            )

            # 7. Notify user
            await self.bot.send_vacancy_card(
                vacancy=vacancy_data,
                application_id=app_id,
                cover_letter=cover_letter,
                match_result=match_result,
                company_info=company_info,
            )
            matched_count += 1

            # Small delay to respect Groq rate limits
            await asyncio.sleep(1)

        # Update last processed message ID
        await self.db.update_last_message_id(username, max_msg_id)
        return matched_count

    async def scan_all_channels(self) -> None:
        """Scan all enabled channels for new vacancies."""
        logger.info("Starting channel scan...")
        channels = await self.db.get_enabled_channels()
        total_matched = 0

        for channel in channels:
            try:
                matched = await self.process_channel(channel)
                total_matched += matched
            except Exception as e:
                logger.error("Error processing @{}: {}", channel["username"], e)

        logger.info("Scan complete. {} new matches found.", total_matched)

    async def run(self) -> None:
        """Start the agent: scheduler + bot polling."""
        await self.init()

        # Schedule periodic scanning
        self.scheduler.add_job(
            self.scan_all_channels,
            "interval",
            minutes=settings.parse_interval_minutes,
            id="channel_scan",
            next_run_time=None,  # don't run immediately, let first scan be manual
        )
        self.scheduler.start()
        logger.info("Scheduler started (interval: {} min)", settings.parse_interval_minutes)

        # Run first scan
        await self.scan_all_channels()

        # Start bot polling (blocks)
        logger.info("Starting Telegram bot polling...")
        await self.bot.app.initialize()
        await self.bot.app.start()
        await self.bot.app.updater.start_polling(drop_pending_updates=True)

        # Keep running
        try:
            while True:
                await asyncio.sleep(3600)
        except (KeyboardInterrupt, SystemExit):
            logger.info("Shutting down...")
        finally:
            await self.bot.app.updater.stop()
            await self.bot.app.stop()
            await self.bot.app.shutdown()
            await self.parser.stop()
            self.scheduler.shutdown()
            logger.info("Goodbye!")
