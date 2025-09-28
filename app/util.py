# util.py

# Model Selection
#modelId = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
modelId = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
modelType = "sonnet" if "sonnet" in modelId else "haiku"


## Manage model configurations for CSV row limits
MODEL_CONFIG = {
    "haiku": {"max_tokens": 4096, "max_rows": 300},
    "sonnet": {"max_tokens": 200000, "max_rows": None}
}

def get_max_rows(model_type: str) -> int:
    """Return max safe row count for CSV based on model type."""
    return MODEL_CONFIG.get(model_type, {}).get("max_rows", 300)


# Logging Util
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)






