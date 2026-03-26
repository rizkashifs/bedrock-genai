"""Chat history formatting helpers (from rag-csv-expert v2)."""
from typing import Any, Dict, List


def _truncate_text(text: str, max_length: int = 200) -> str:
    text = text.strip()
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


def truncate_history(history: List[Dict[str, Any]], max_user_turns: int = 5) -> str:
    """
    Format the last N conversation turns into a plain-text string for LLM context.
    Assistant responses are truncated to 200 chars to keep prompts compact.
    """
    if not history:
        return ""

    recent = history[-max_user_turns:]
    lines: List[str] = []

    for turn in recent:
        user_query = turn.get("user", "").strip()
        assistant_text = turn.get("assistant", "").strip()
        if user_query:
            lines.append(f"User: {user_query}")
        if assistant_text:
            lines.append(f"Assistant: {_truncate_text(assistant_text)}")

    return "\n".join(lines)
