#!/usr/bin/env python3
"""
Server worker — runs on VPS.
Handles: Telegram parsing, SQLite DB, bot notifications, HTTP API for local worker.
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aiohttp import web
from loguru import logger
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from config.settings import settings
from src.database.models import Database, init_db
from src.parser.telegram_parser import TelegramParser
from src.bot.notification_bot import NotificationBot

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | <cyan>{name}</cyan> — <level>{message}</level>",
    level="INFO",
)
logger.add("data/agent.log", rotation="10 MB", retention="7 days", level="DEBUG")


class ServerWorker:
    def __init__(self):
        self.db = Database(settings.db_path)
        self.parser = TelegramParser(
            api_id=settings.telegram_api_id,
            api_hash=settings.telegram_api_hash,
            phone=settings.telegram_phone,
            session_path=settings.session_path,
        )
        self.bot = NotificationBot(
            token=settings.telegram_bot_token,
            owner_id=settings.telegram_owner_id,
            db=self.db,
        )
        self.scheduler = AsyncIOScheduler()

    async def init(self):
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        await init_db(settings.db_path)

        channels = settings.load_channels()
        for ch in channels:
            await self.db.upsert_channel(
                username=ch["username"],
                name=ch.get("name"),
                category=ch.get("category"),
                enabled=ch.get("enabled", True),
            )
        logger.info("Synced {} channels from config", len(channels))

        await self.parser.start()
        self.bot.build()

    async def parse_channels(self):
        """Parse all channels, save raw messages to DB with status='new'."""
        logger.info("Parsing channels...")
        channels = await self.db.get_enabled_channels()
        total_new = 0

        for channel in channels:
            try:
                username = channel["username"]
                min_id = channel.get("last_message_id", 0) or 0

                messages = await self.parser.get_new_messages(username, min_id=min_id)
                if not messages:
                    continue

                max_msg_id = max(m["message_id"] for m in messages)

                for msg in messages:
                    vacancy_id = await self.db.insert_vacancy(
                        channel_username=username,
                        message_id=msg["message_id"],
                        raw_text=msg["text"],
                    )
                    if vacancy_id:
                        total_new += 1

                await self.db.update_last_message_id(username, max_msg_id)
            except Exception as e:
                logger.error("Error parsing @{}: {}", channel["username"], e)

        logger.info("Parsing done. {} new messages saved.", total_new)

    # --- HTTP API for local worker ---

    async def handle_get_unprocessed(self, request):
        """GET /api/vacancies/unprocessed — return raw vacancies for Groq analysis."""
        vacancies = await self.db.get_new_vacancies()
        return web.json_response(vacancies)

    async def handle_update_vacancy(self, request):
        """POST /api/vacancies/<id>/result — receive Groq analysis results."""
        vacancy_id = int(request.match_info["id"])
        data = await request.json()

        # Update vacancy with extracted data
        if data.get("vacancy_data"):
            vd = data["vacancy_data"]
            async with self.db._connect() as db:
                await db.execute(
                    """UPDATE vacancies SET
                       title=?, company=?, location=?, salary=?,
                       skills=?, requirements=?, remote=?, contact=?, source_url=?
                       WHERE id=?""",
                    (vd.get("title"), vd.get("company"), vd.get("location"),
                     vd.get("salary"), vd.get("skills"), vd.get("requirements"),
                     vd.get("remote"), vd.get("contact"), vd.get("source_url"),
                     vacancy_id)
                )
                await db.commit()

        # Update match score
        if data.get("match_result"):
            mr = data["match_result"]
            score = mr.get("score", 0)
            await self.db.update_vacancy_match(vacancy_id, score, mr.get("reason", ""))

            if score < settings.match_threshold:
                return web.json_response({"status": "below_threshold", "score": score})

        # Save application if above threshold
        if data.get("cover_letter"):
            app_id = await self.db.create_application(
                vacancy_id=vacancy_id,
                cover_letter=data["cover_letter"],
                company_info=json.dumps(data.get("company_info", {}), ensure_ascii=False),
            )

            # Send notification via bot
            await self.bot.send_vacancy_card(
                vacancy=data.get("vacancy_data", {}),
                application_id=app_id,
                cover_letter=data["cover_letter"],
                match_result=data.get("match_result", {}),
                company_info=data.get("company_info", {}),
            )
            return web.json_response({"status": "notified", "application_id": app_id})

        return web.json_response({"status": "updated"})

    async def handle_health(self, request):
        return web.json_response({"status": "ok", "channels": len(await self.db.get_enabled_channels())})

    async def run(self):
        await self.init()

        # Schedule parsing
        self.scheduler.add_job(
            self.parse_channels,
            "interval",
            minutes=settings.parse_interval_minutes,
            id="channel_parse",
            next_run_time=None,
        )
        self.scheduler.start()
        logger.info("Scheduler started (interval: {} min)", settings.parse_interval_minutes)

        # First parse
        await self.parse_channels()

        # Start bot polling in background
        await self.bot.app.initialize()
        await self.bot.app.start()
        await self.bot.app.updater.start_polling(drop_pending_updates=True)
        logger.info("Telegram bot polling started")

        # Start HTTP API
        app = web.Application()
        app.router.add_get("/api/health", self.handle_health)
        app.router.add_get("/api/vacancies/unprocessed", self.handle_get_unprocessed)
        app.router.add_post("/api/vacancies/{id}/result", self.handle_update_vacancy)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 8642)
        await site.start()
        logger.info("HTTP API listening on 127.0.0.1:8642")

        try:
            while True:
                await asyncio.sleep(3600)
        except (KeyboardInterrupt, SystemExit):
            logger.info("Shutting down...")
        finally:
            await runner.cleanup()
            await self.bot.app.updater.stop()
            await self.bot.app.stop()
            await self.bot.app.shutdown()
            await self.parser.stop()
            self.scheduler.shutdown()


if __name__ == "__main__":
    asyncio.run(ServerWorker().run())
