import asyncio
import json
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    MessageHandler, ContextTypes, filters,
)
from loguru import logger
from src.database.models import Database


class NotificationBot:
    def __init__(self, token: str, owner_id: int, db: Database):
        self.token = token
        self.owner_id = owner_id
        self.db = db
        self.app: Application | None = None
        self._pending_comment_for: dict[int, int] = {}
        self._digest_buffer: list[dict] = []
        self._digest_lock = asyncio.Lock()
        self.DIGEST_SIZE = 5  # max vacancies per digest
        self.DIGEST_INTERVAL = 300  # seconds (5 min)
        self._digest_task: asyncio.Task | None = None

    def build(self) -> Application:
        self.app = Application.builder().token(self.token).build()
        self.app.add_handler(CommandHandler("start", self._cmd_start))
        self.app.add_handler(CommandHandler("stats", self._cmd_stats))
        self.app.add_handler(CallbackQueryHandler(self._on_callback))
        self.app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_text)
        )
        return self.app

    def start_digest_loop(self):
        """Start background digest sender."""
        if self._digest_task is None or self._digest_task.done():
            self._digest_task = asyncio.create_task(self._digest_loop())
            logger.info("Digest loop started (interval: {}s, batch: {})",
                        self.DIGEST_INTERVAL, self.DIGEST_SIZE)

    async def _digest_loop(self):
        """Periodically flush digest buffer."""
        while True:
            await asyncio.sleep(self.DIGEST_INTERVAL)
            await self._flush_digest()

    async def add_to_digest(self, vacancy: dict, application_id: int,
                            cover_letter: str, match_result: dict,
                            company_info: dict) -> None:
        """Add vacancy to digest buffer. Flushes when full."""
        async with self._digest_lock:
            self._digest_buffer.append({
                "vacancy": vacancy,
                "application_id": application_id,
                "cover_letter": cover_letter,
                "match_result": match_result,
                "company_info": company_info,
            })
            if len(self._digest_buffer) >= self.DIGEST_SIZE:
                await self._flush_digest_locked()

    async def _flush_digest(self):
        async with self._digest_lock:
            await self._flush_digest_locked()

    async def _flush_digest_locked(self):
        """Send buffered vacancies as digest. Must be called with lock held."""
        if not self._digest_buffer:
            return

        items = self._digest_buffer[:]
        self._digest_buffer.clear()

        # Header
        n = len(items)
        text = f"📋 <b>Подборка: {n} вакансий</b>\n{'─' * 30}\n\n"

        # Build compact cards
        for i, item in enumerate(items, 1):
            v = item["vacancy"]
            mr = item["match_result"]
            hook = mr.get("hook", {})
            top5 = mr.get("top5", {})
            app_id = item["application_id"]
            score = mr.get("score", 0)
            rec = mr.get("recommendation", "")
            rec_emoji = {"strong_apply": "🔥", "apply": "✅"}.get(rec, "✅")

            text += f"<b>{i}. {rec_emoji} {v.get('title', '?')}</b>\n"
            text += f"   🏢 {v.get('company', '?')}"
            if v.get("salary"):
                text += f" · 💰 {v['salary']}"
            if v.get("remote"):
                text += " · 🏠"
            text += "\n"

            # Hook — what caught attention
            if hook.get("hook"):
                text += f"   🪝 {hook['hook']}\n"

            # Domain & values match
            if hook.get("domain_match"):
                text += f"   🎯 Домен: {hook['domain_match']}\n"

            # Top-5 edge
            if top5 and top5.get("his_edge"):
                text += f"   ⚔️ Преимущество: {top5['his_edge']}\n"

            # Percentile
            if top5 and top5.get("percentile"):
                text += f"   🏆 Top-{top5['percentile']} из 100\n"

            # Contact
            if v.get("contact"):
                text += f"   📬 {v['contact']}\n"
            if v.get("source_url"):
                text += f"   🔗 {v['source_url']}\n"

            text += "\n"

        # Trim to Telegram limit
        if len(text) > 4000:
            text = text[:3990] + "\n..."

        # Buttons for each vacancy
        buttons = []
        for i, item in enumerate(items, 1):
            app_id = item["application_id"]
            title = item["vacancy"].get("title", "?")[:20]
            buttons.append([
                InlineKeyboardButton(f"✅ {i}. {title}", callback_data=f"apply:{app_id}"),
                InlineKeyboardButton("❌", callback_data=f"skip:{app_id}"),
                InlineKeyboardButton("📖", callback_data=f"detail:{app_id}"),
            ])

        keyboard = InlineKeyboardMarkup(buttons)

        await self.app.bot.send_message(
            chat_id=self.owner_id,
            text=text,
            parse_mode="HTML",
            reply_markup=keyboard,
            disable_web_page_preview=True,
        )
        logger.info("Sent digest with {} vacancies", n)

    # Legacy method — now routes to digest
    async def send_vacancy_card(self, vacancy: dict, application_id: int,
                                cover_letter: str, match_result: dict,
                                company_info: dict) -> None:
        await self.add_to_digest(vacancy, application_id, cover_letter,
                                 match_result, company_info)

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🤖 <b>AI Job Seeker Bot</b>\n\n"
            "Присылаю подборки подходящих вакансий.\n\n"
            "Команды:\n"
            "/stats — статистика\n\n"
            "Кнопки под подборкой:\n"
            "✅ — откликнуться\n"
            "❌ — пропустить\n"
            "📖 — подробнее (интро + компания)",
            parse_mode="HTML",
        )

    async def _cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.effective_user.id != self.owner_id:
            return
        import aiosqlite
        async with aiosqlite.connect(self.db.db_path) as db:
            cur = await db.execute("SELECT COUNT(*) FROM vacancies")
            total = (await cur.fetchone())[0]

            cur = await db.execute("SELECT COUNT(*) FROM vacancies WHERE status = 'new'")
            pending = (await cur.fetchone())[0]

            cur = await db.execute("SELECT COUNT(*) FROM vacancies WHERE match_score > 0")
            analyzed = (await cur.fetchone())[0]

            cur = await db.execute("SELECT COUNT(*) FROM vacancies WHERE match_score >= 70")
            matched = (await cur.fetchone())[0]

            cur = await db.execute("SELECT COUNT(*) FROM applications")
            total_apps = (await cur.fetchone())[0]

            cur = await db.execute("SELECT COUNT(*) FROM applications WHERE user_decision = 'apply'")
            applied = (await cur.fetchone())[0]

            cur = await db.execute("SELECT COUNT(*) FROM applications WHERE user_decision = 'skip'")
            skipped = (await cur.fetchone())[0]

            cur = await db.execute("SELECT COUNT(*) FROM channels WHERE enabled = 1")
            channels_active = (await cur.fetchone())[0]

            cur = await db.execute("SELECT COUNT(*) FROM channels WHERE enabled = 0")
            channels_disabled = (await cur.fetchone())[0]

            # Top channels by vacancy count
            cur = await db.execute("""
                SELECT channel_username, COUNT(*) as cnt,
                       SUM(CASE WHEN match_score >= 70 THEN 1 ELSE 0 END) as good
                FROM vacancies WHERE match_score IS NOT NULL
                GROUP BY channel_username ORDER BY good DESC LIMIT 5
            """)
            top_channels = await cur.fetchall()

        text = (
            f"📊 <b>Статистика</b>\n\n"
            f"<b>Вакансии:</b>\n"
            f"  📥 Всего спарсено: {total}\n"
            f"  ⏳ Ожидают анализа: {pending}\n"
            f"  🔍 Проанализировано: {analyzed}\n"
            f"  🎯 Прошли фильтр (score≥70): {matched}\n\n"
            f"<b>Отклики:</b>\n"
            f"  📨 Отправлено подборок: {total_apps}\n"
            f"  ✅ Откликнулся: {applied}\n"
            f"  ❌ Пропустил: {skipped}\n\n"
            f"<b>Каналы:</b>\n"
            f"  ✅ Активных: {channels_active}\n"
            f"  ⛔ Отключено: {channels_disabled}\n"
        )

        if top_channels:
            text += "\n<b>Топ каналов по релевантности:</b>\n"
            for ch in top_channels:
                text += f"  @{ch[0]}: {ch[2]}/{ch[1]} подходящих\n"

        await update.message.reply_text(text, parse_mode="HTML")

    async def _on_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        if query.from_user.id != self.owner_id:
            await query.answer("Нет доступа")
            return

        await query.answer()
        action, app_id_str = query.data.split(":", 1)
        app_id = int(app_id_str)

        if action == "apply":
            await self.db.update_application_decision(app_id, "apply")
            app = await self._get_app_with_vacancy(app_id)
            title = app.get("title", "вакансию") if app else "вакансию"
            await query.message.reply_text(
                f"✅ Отклик на «{title}» подтверждён!\n"
                f"Нажми 📖 чтобы увидеть адаптированное интро."
            )

        elif action == "skip":
            await self.db.update_application_decision(app_id, "skip")
            await query.message.reply_text("❌ Пропущено")

        elif action == "detail":
            # Show full details: cover letter + company info
            app_data = await self._get_full_application(app_id)
            if app_data:
                text = ""
                if app_data.get("company_info"):
                    try:
                        ci = json.loads(app_data["company_info"])
                        if ci.get("summary"):
                            text += f"🏢 <b>О компании:</b>\n{ci['summary']}\n\n"
                    except (json.JSONDecodeError, TypeError):
                        pass
                if app_data.get("cover_letter"):
                    text += f"✉️ <b>Адаптированное интро:</b>\n<i>{app_data['cover_letter']}</i>"
                if text:
                    await query.message.reply_text(
                        text[:4096], parse_mode="HTML"
                    )
                else:
                    await query.message.reply_text("Детали пока не готовы")
            else:
                await query.message.reply_text("Не найдено")

        elif action == "comment":
            self._pending_comment_for[query.from_user.id] = app_id
            await query.message.reply_text("✏️ Напиши комментарий:")

    async def _on_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if user_id != self.owner_id:
            return
        if user_id in self._pending_comment_for:
            app_id = self._pending_comment_for.pop(user_id)
            await self.db.update_application_decision(app_id, "feedback", update.message.text)
            await update.message.reply_text(f"📝 Учту: «{update.message.text}»")
        else:
            await update.message.reply_text("Используй кнопки под подборкой или /stats")

    async def _get_app_with_vacancy(self, app_id: int) -> dict | None:
        import aiosqlite
        async with aiosqlite.connect(self.db.db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                """SELECT v.* FROM applications a
                   JOIN vacancies v ON a.vacancy_id = v.id
                   WHERE a.id = ?""", (app_id,)
            )
            row = await cur.fetchone()
            return dict(row) if row else None

    async def _get_full_application(self, app_id: int) -> dict | None:
        import aiosqlite
        async with aiosqlite.connect(self.db.db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT * FROM applications WHERE id = ?", (app_id,)
            )
            row = await cur.fetchone()
            return dict(row) if row else None
