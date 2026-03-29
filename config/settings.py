import json
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # Telegram Userbot (Telethon)
    telegram_api_id: int = Field(..., description="Telegram API ID from my.telegram.org")
    telegram_api_hash: str = Field(..., description="Telegram API Hash")
    telegram_phone: str = Field(..., description="Phone number for Telethon auth")

    # Telegram Bot
    telegram_bot_token: str = Field(..., description="Bot token from @BotFather")
    telegram_owner_id: int = Field(..., description="Your Telegram user ID for notifications")

    # LLM Provider
    llm_provider: str = Field(default="cerebras", description="Primary LLM provider: cerebras, groq")
    llm_api_key: str = Field(default="", description="Primary LLM API key")
    llm_model: str = Field(default="llama3.1-8b", description="Primary LLM model")

    # Groq (fallback)
    groq_api_key: str = Field(default="", description="Groq API key (fallback)")
    groq_model: str = Field(default="llama-3.3-70b-versatile", description="Groq model name")

    # Matching
    match_threshold: int = Field(default=60, ge=0, le=100, description="Minimum match score (0-100)")

    # Scheduler
    parse_interval_minutes: int = Field(default=10, ge=1, description="Interval between channel scans")

    # Paths
    data_dir: Path = Field(default=BASE_DIR / "data")
    db_path: Path = Field(default=BASE_DIR / "data" / "jobs.db")
    resume_path: Path = Field(default=BASE_DIR / "data" / "resume.pdf")
    profile_path: Path = Field(default=BASE_DIR / "data" / "profile.json")
    channels_path: Path = Field(default=BASE_DIR / "config" / "channels.json")
    session_path: Path = Field(default=BASE_DIR / "data" / "userbot_session")

    model_config = {
        "env_file": str(BASE_DIR / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    def load_channels(self) -> list[dict]:
        """Load channel list from JSON config."""
        with open(self.channels_path, "r", encoding="utf-8") as f:
            return json.load(f)["channels"]

    def load_profile(self) -> dict:
        """Load candidate profile (LinkedIn + intro)."""
        with open(self.profile_path, "r", encoding="utf-8") as f:
            return json.load(f)


settings = Settings()
