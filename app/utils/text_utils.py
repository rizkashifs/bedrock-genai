"""Text normalisation helpers (from rag-pdf-expert)."""
import re
from typing import List

WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and trim."""
    return WHITESPACE_RE.sub(" ", text).strip()


def split_into_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter on .!? boundaries."""
    raw = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in raw if s.strip()]
