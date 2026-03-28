#!/usr/bin/env python3
"""AI Job Seeker Agent — entry point."""
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from loguru import logger
from src.main import JobSeekerAgent

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | <cyan>{name}</cyan> — <level>{message}</level>",
    level="INFO",
)
logger.add(
    "data/agent.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
)


async def main():
    agent = JobSeekerAgent()
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
