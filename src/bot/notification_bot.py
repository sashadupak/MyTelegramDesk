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
        self._pending_comment_for: dict[int, int] = {}  # user_id -> application_id

    def build(self) -> Application:
        """Build the bot application with handlers."""
        self.app = Application.builder().token(self.token).build()
        self.app.add_handler(CommandHandler("start", self._cmd_start))
        self.app.add_handler(CommandHandler("stats", self._cmd_stats))
        self.app.add_handler(CallbackQueryHandler(self._on_callback))
        self.app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_text)
        )
        return self.app

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "AI Job Seeker Bot\n\n"
            "Я буду присылать вам подходящие вакансии с адаптированным интро.\n\n"
            "Команды:\n"
            "/stats — статистика по вакансиям"
        )

    async def _cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.effective_user.id != self.owner_id:
            return
        # Simple stats query
        import aiosqlite
        async with aiosqlite.connect(self.db.db_path) as db:
            cur = await db.execute("SELECT COUNT(*) FROM vacancies")
            total = (await cur.fetchone())[0]
            cur = await db.execute("SELECT COUNT(*) FROM vacancies WHERE status = 'matched'")
            matched = (await cur.fetchone())[0]
            cur = await db.execute("SELECT COUNT(*) FROM applications WHERE user_decision = 'apply'")
            applied = (await cur.fetchone())[0]
            cur = await db.execute("SELECT COUNT(*) FROM applications WHERE user_decision = 'skip'")
            skipped = (await cur.fetchone())[0]

        await update.message.reply_text(
            f"Статистика:\n"
            f"Всего вакансий: {total}\n"
            f"Подходящих: {matched}\n"
            f"Откликнулись: {applied}\n"
            f"Пропущено: {skipped}"
        )

    async def send_vacancy_card(self, vacancy: dict, application_id: int,
                                cover_letter: str, match_result: dict,
                                company_info: dict) -> None:
        """Send a vacancy card to the owner with action buttons."""
        score = match_result.get("score", 0)
        rec = match_result.get("recommendation", "")
        rec_emoji = {"strong_apply": "🔥", "apply": "✅", "maybe": "🤔", "skip": "⛔"}.get(rec, "❓")
        score_bar = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"

        # Venture context
        vf = match_result.get("venture_filter", {})
        env_type = vf.get("environment_type", "?")

        text = (
            f"{score_bar} <b>Score: {score}/100</b> {rec_emoji} {rec}\n"
            f"🏷 {env_type}\n\n"
            f"<b>{vacancy.get('title', 'Без названия')}</b>\n"
            f"🏢 {vacancy.get('company', 'Компания не указана')}\n"
        )
        if vacancy.get("location"):
            text += f"📍 {vacancy['location']}\n"
        if vacancy.get("salary"):
            text += f"💰 {vacancy['salary']}\n"
        if vacancy.get("remote"):
            text += "🏠 Удалёнка\n"
        if vacancy.get("skills"):
            text += f"🛠 {vacancy['skills']}\n"

        # Dimensions breakdown
        dims = match_result.get("dimensions", {})
        if dims:
            text += "\n<b>Оценка по измерениям:</b>\n"
            dim_names = {
                "role_fit": "🎯 Role Fit",
                "skills_match": "🛠 Skills",
                "seniority_fit": "📊 Seniority",
                "competitive_edge": "⚔️ Edge",
                "growth_potential": "🚀 Growth",
            }
            for key, label in dim_names.items():
                d = dims.get(key, {})
                d_score = d.get("score", 0)
                bar = "█" * (d_score // 4) + "░" * (5 - d_score // 4)
                text += f"  {label}: {bar} {d_score}/20\n"

        # Strengths & gaps
        strengths = match_result.get("strengths", [])
        gaps = match_result.get("gaps", [])
        if strengths:
            text += f"\n<b>💪 Сильные:</b> {', '.join(strengths[:3])}\n"
        if gaps:
            text += f"<b>⚠️ Пробелы:</b> {', '.join(gaps[:3])}\n"

        text += (
            f"\n<b>Вердикт:</b>\n{match_result.get('reason', 'N/A')}\n"
            f"\n<b>О компании:</b>\n{company_info.get('summary', 'N/A')}\n"
            f"\n<b>Адаптированное интро:</b>\n<i>{cover_letter}</i>"
        )

        if vacancy.get("contact"):
            text += f"\n\n📬 Контакт: {vacancy['contact']}"
        if vacancy.get("source_url"):
            text += f"\n🔗 {vacancy['source_url']}"

        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("✅ Откликнуться", callback_data=f"apply:{application_id}"),
                InlineKeyboardButton("❌ Пропустить", callback_data=f"skip:{application_id}"),
            ],
            [
                InlineKeyboardButton("✏️ Комментарий", callback_data=f"comment:{application_id}"),
            ],
        ])

        await self.app.bot.send_message(
            chat_id=self.owner_id,
            text=text[:4096],
            parse_mode="HTML",
            reply_markup=keyboard,
        )
        logger.info("Sent vacancy card for application #{}", application_id)

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
            vacancy_title = app.get("title", "вакансию") if app else "вакансию"
            await query.edit_message_reply_markup(reply_markup=None)
            await query.message.reply_text(
                f"✅ Отлично! Отклик на «{vacancy_title}» подтверждён.\n"
                f"Используйте адаптированное интро выше для отклика."
            )
            logger.info("User approved application #{}", app_id)

        elif action == "skip":
            await self.db.update_application_decision(app_id, "skip")
            await query.edit_message_reply_markup(reply_markup=None)
            await query.message.reply_text("❌ Вакансия пропущена.")
            logger.info("User skipped application #{}", app_id)

        elif action == "comment":
            self._pending_comment_for[query.from_user.id] = app_id
            await query.message.reply_text(
                "✏️ Напишите комментарий — что скорректировать в поиске?"
            )

    async def _on_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if user_id != self.owner_id:
            return

        if user_id in self._pending_comment_for:
            app_id = self._pending_comment_for.pop(user_id)
            comment = update.message.text
            await self.db.update_application_decision(app_id, "feedback", comment)
            await update.message.reply_text(
                f"📝 Комментарий сохранён. Учту при следующих подборках:\n«{comment}»"
            )
            logger.info("User left feedback on application #{}: {}", app_id, comment)
        else:
            await update.message.reply_text(
                "Используйте кнопки под вакансиями или команду /stats"
            )

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
