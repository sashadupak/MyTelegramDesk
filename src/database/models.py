import aiosqlite
from datetime import datetime
from pathlib import Path
from loguru import logger


async def init_db(db_path: Path) -> None:
    """Create database tables if they don't exist."""
    async with aiosqlite.connect(db_path) as db:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS channels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                name TEXT,
                category TEXT,
                enabled INTEGER DEFAULT 1,
                last_message_id INTEGER DEFAULT 0,
                consecutive_non_vacancy INTEGER DEFAULT 0,
                total_parsed INTEGER DEFAULT 0,
                total_vacancies INTEGER DEFAULT 0,
                disabled_reason TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS vacancies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_username TEXT NOT NULL,
                message_id INTEGER NOT NULL,
                raw_text TEXT NOT NULL,
                title TEXT,
                company TEXT,
                location TEXT,
                salary TEXT,
                skills TEXT,
                requirements TEXT,
                remote INTEGER,
                contact TEXT,
                source_url TEXT,
                match_score INTEGER,
                match_reason TEXT,
                status TEXT DEFAULT 'new',
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(channel_username, message_id)
            );

            CREATE TABLE IF NOT EXISTS applications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vacancy_id INTEGER NOT NULL REFERENCES vacancies(id),
                cover_letter TEXT,
                company_info TEXT,
                user_decision TEXT,
                user_comment TEXT,
                decided_at TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_vacancies_status ON vacancies(status);
            CREATE INDEX IF NOT EXISTS idx_vacancies_score ON vacancies(match_score);
        """)

        # Migrate: add new columns if missing
        for col, default in [
            ("consecutive_non_vacancy", "0"),
            ("total_parsed", "0"),
            ("total_vacancies", "0"),
            ("disabled_reason", "NULL"),
        ]:
            try:
                await db.execute(f"ALTER TABLE channels ADD COLUMN {col} DEFAULT {default}")
                await db.commit()
            except Exception:
                pass  # column already exists

        logger.info("Database initialized at {}", db_path)


class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def _connect(self) -> aiosqlite.Connection:
        return aiosqlite.connect(self.db_path)

    # --- Channels ---

    async def upsert_channel(self, username: str, name: str = None,
                             category: str = None, enabled: bool = True) -> None:
        async with self._connect() as db:
            await db.execute(
                """INSERT INTO channels (username, name, category, enabled)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(username) DO UPDATE SET
                     name=excluded.name, category=excluded.category, enabled=excluded.enabled""",
                (username, name, category, int(enabled))
            )
            await db.commit()

    async def get_enabled_channels(self) -> list[dict]:
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM channels WHERE enabled = 1"
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def update_last_message_id(self, username: str, message_id: int) -> None:
        async with self._connect() as db:
            await db.execute(
                "UPDATE channels SET last_message_id = ? WHERE username = ?",
                (message_id, username)
            )
            await db.commit()

    async def record_vacancy_hit(self, username: str) -> None:
        """Message was a vacancy — reset consecutive counter, bump totals."""
        async with self._connect() as db:
            await db.execute(
                """UPDATE channels SET
                   consecutive_non_vacancy = 0,
                   total_parsed = total_parsed + 1,
                   total_vacancies = total_vacancies + 1
                   WHERE username = ?""",
                (username,)
            )
            await db.commit()

    async def record_non_vacancy(self, username: str, threshold: int = 50) -> bool:
        """Message was NOT a vacancy. Returns True if channel should be disabled."""
        async with self._connect() as db:
            await db.execute(
                """UPDATE channels SET
                   consecutive_non_vacancy = consecutive_non_vacancy + 1,
                   total_parsed = total_parsed + 1
                   WHERE username = ?""",
                (username,)
            )
            await db.commit()

            cursor = await db.execute(
                "SELECT consecutive_non_vacancy, total_parsed, total_vacancies FROM channels WHERE username = ?",
                (username,)
            )
            row = await cursor.fetchone()
            if row and row[0] >= threshold:
                await db.execute(
                    """UPDATE channels SET enabled = 0,
                       disabled_reason = ? WHERE username = ?""",
                    (f"Auto-disabled: {row[0]} consecutive non-vacancies "
                     f"(total: {row[1]} parsed, {row[2]} vacancies)", username)
                )
                await db.commit()
                return True
        return False

    # --- Vacancies ---

    async def insert_vacancy(self, channel_username: str, message_id: int,
                             raw_text: str, **kwargs) -> int | None:
        """Insert vacancy, return its ID. Returns None if duplicate."""
        async with self._connect() as db:
            try:
                cursor = await db.execute(
                    """INSERT INTO vacancies
                       (channel_username, message_id, raw_text, title, company,
                        location, salary, skills, requirements, remote, contact, source_url)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (channel_username, message_id, raw_text,
                     kwargs.get("title"), kwargs.get("company"),
                     kwargs.get("location"), kwargs.get("salary"),
                     kwargs.get("skills"), kwargs.get("requirements"),
                     kwargs.get("remote"), kwargs.get("contact"),
                     kwargs.get("source_url"))
                )
                await db.commit()
                return cursor.lastrowid
            except aiosqlite.IntegrityError:
                return None

    async def update_vacancy_match(self, vacancy_id: int, score: int, reason: str) -> None:
        async with self._connect() as db:
            await db.execute(
                "UPDATE vacancies SET match_score = ?, match_reason = ?, status = 'matched' WHERE id = ?",
                (score, reason, vacancy_id)
            )
            await db.commit()

    async def update_vacancy_status(self, vacancy_id: int, status: str) -> None:
        async with self._connect() as db:
            await db.execute(
                "UPDATE vacancies SET status = ? WHERE id = ?",
                (status, vacancy_id)
            )
            await db.commit()

    async def get_vacancy(self, vacancy_id: int) -> dict | None:
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM vacancies WHERE id = ?", (vacancy_id,)
            )
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def get_new_vacancies(self) -> list[dict]:
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM vacancies WHERE status = 'new' ORDER BY created_at"
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    # --- Applications ---

    async def create_application(self, vacancy_id: int, cover_letter: str,
                                 company_info: str) -> int:
        async with self._connect() as db:
            cursor = await db.execute(
                """INSERT INTO applications (vacancy_id, cover_letter, company_info)
                   VALUES (?, ?, ?)""",
                (vacancy_id, cover_letter, company_info)
            )
            await db.commit()
            return cursor.lastrowid

    async def update_application_decision(self, application_id: int,
                                          decision: str, comment: str = None) -> None:
        async with self._connect() as db:
            await db.execute(
                """UPDATE applications
                   SET user_decision = ?, user_comment = ?, decided_at = ?
                   WHERE id = ?""",
                (decision, comment, datetime.now().isoformat(), application_id)
            )
            await db.commit()

    async def get_application_by_vacancy(self, vacancy_id: int) -> dict | None:
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM applications WHERE vacancy_id = ?", (vacancy_id,)
            )
            row = await cursor.fetchone()
            return dict(row) if row else None
