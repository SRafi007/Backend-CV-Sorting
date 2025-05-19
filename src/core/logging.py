from loguru import logger
import sys
from pathlib import Path

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)

# Remove default logger to customize it
logger.remove()

# Add stdout (console) logging
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

# Add file logging
logger.add(
    log_dir / "app.log",
    rotation="1 week",  # Rotate logs weekly
    retention="1 month",  # Keep logs for 1 month
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
    level="DEBUG"
)
