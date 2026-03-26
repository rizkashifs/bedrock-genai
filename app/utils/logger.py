"""
Shared logger for the entire application.
Import this instead of creating module-level loggers.
"""
import logging
from pathlib import Path

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = Path(__file__).resolve().parents[2] / "rag_system.log"

# Configure root only once
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(),
        ],
    )

logger = logging.getLogger("bedrock_genai")
