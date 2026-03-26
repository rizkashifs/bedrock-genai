# util.py — re-exports for backwards compatibility.
# New code should import directly from app.config.settings or app.utils.logger.

from app.config.settings import settings
from app.utils.logger import logger  # noqa: F401  (re-exported)

# ── Model identifiers (kept for backwards compatibility) ───────────────────
modelId: str = settings.bedrock_model_id
modelType: str = settings.model_type

# ── Per-model limits ───────────────────────────────────────────────────────
MODEL_CONFIG = {
    "haiku": {"max_tokens": 4_096, "max_rows": 300},
    "sonnet": {"max_tokens": 200_000, "max_rows": None},
}


def get_max_rows(model_type: str) -> int:
    """Return max safe CSV row count for the given model type."""
    return MODEL_CONFIG.get(model_type, {}).get("max_rows", 300)
