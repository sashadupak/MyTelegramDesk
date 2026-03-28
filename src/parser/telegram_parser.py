from telethon import TelegramClient
from telethon.tl.types import Message
from loguru import logger
from pathlib import Path


class TelegramParser:
    def __init__(self, api_id: int, api_hash: str, phone: str, session_path: Path):
        self.client = TelegramClient(
            str(session_path), api_id, api_hash
        )
        self.phone = phone

    async def start(self) -> None:
        """Start the Telethon client."""
        await self.client.start(phone=self.phone)
        me = await self.client.get_me()
        logger.info("Telethon connected as {} ({})", me.first_name, me.id)

    async def stop(self) -> None:
        """Disconnect the client."""
        await self.client.disconnect()

    async def get_new_messages(self, channel_username: str,
                               min_id: int = 0, limit: int = 50) -> list[dict]:
        """
        Fetch new messages from a channel since min_id.
        Returns list of dicts with message data.
        """
        messages = []
        try:
            entity = await self.client.get_entity(channel_username)
            async for msg in self.client.iter_messages(
                entity, limit=limit, min_id=min_id
            ):
                if not isinstance(msg, Message) or not msg.text:
                    continue
                messages.append({
                    "message_id": msg.id,
                    "text": msg.text,
                    "date": msg.date.isoformat(),
                    "channel": channel_username,
                    "url": f"https://t.me/{channel_username}/{msg.id}",
                })
            logger.debug("Fetched {} messages from @{} (min_id={})",
                         len(messages), channel_username, min_id)
        except Exception as e:
            logger.error("Error fetching @{}: {}", channel_username, e)
        return messages
