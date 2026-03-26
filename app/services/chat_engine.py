"""
Chat Engine — multi-turn conversation management via AWS Bedrock.

Supports:
  - In-memory history (always available)
  - Optional DynamoDB persistence (requires external dynamo_chat_storage layer)
  - Two output modes: "api" (structured JSON) and "chat" (conversational)
  - Four feature prompts: qna, docComparison, search, codeReview
"""

import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from app.models.bedrock_client import bedrock_client
from app.utils.logger import logger

ModeType = Literal["api", "chat"]

# ── DynamoDB storage (optional external dependency) ────────────────────────
try:
    from dynamo_chat_storage import DynamoChatStorage  # type: ignore
    DYNAMO_AVAILABLE = True
except ImportError:
    DYNAMO_AVAILABLE = False
    logger.warning("dynamo_chat_storage not available — running in memory-only mode.")


# ── System-prompt cache ────────────────────────────────────────────────────
_SYSTEM_PROMPTS: Optional[Dict[str, str]] = None

_PROMPTS_FILE = os.path.join(os.path.dirname(__file__), "..", "prompts", "system_prompts.json")


def load_system_prompts() -> Dict[str, str]:
    global _SYSTEM_PROMPTS
    try:
        with open(_PROMPTS_FILE, encoding="utf-8") as f:
            _SYSTEM_PROMPTS = json.load(f)
        return _SYSTEM_PROMPTS
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Required system prompts file not found at: {_PROMPTS_FILE}"
        )
    except json.JSONDecodeError as exc:
        logger.error("Error parsing system_prompts.json: %s", exc)
        return _get_default_prompts()


def _get_default_prompts() -> Dict[str, str]:
    return {
        "docComparison": "You are an expert document analysis assistant specialised in comparing documents.",
        "qna": "You are a knowledgeable Q&A assistant that provides accurate and comprehensive answers.",
        "search": "You are a search assistant that helps users find relevant information efficiently.",
        "codeReview": "You are an expert code reviewer that analyses code for bugs, best practices, and improvements.",
    }


def get_system_prompts() -> Dict[str, str]:
    global _SYSTEM_PROMPTS
    if _SYSTEM_PROMPTS is None:
        _SYSTEM_PROMPTS = load_system_prompts()
    return _SYSTEM_PROMPTS


def get_system_prompt(feature: str, mode: ModeType = "api") -> str:
    prompts = get_system_prompts()
    base = prompts.get(feature, prompts.get("qna", "You are a helpful AI assistant."))
    if mode == "api":
        return f"{base}\n\n{_get_api_output_format(feature)}"
    return f"{base}\n\nProvide responses in a natural, conversational format."


def _get_api_output_format(feature: str) -> str:
    formats: Dict[str, str] = {
        "docComparison": (
            'OUTPUT FORMAT: Return valid JSON:\n'
            '{"summary":"...","differences":[{"type":"addition|deletion|modification",'
            '"location":"...","description":"...","importance":"high|medium|low"}],'
            '"similarities":["..."],"recommendations":["..."]}'
        ),
        "qna": (
            'OUTPUT FORMAT: Return valid JSON:\n'
            '{"answer":"...","confidence":"high|medium|low","sources":["..."],'
            '"related_topics":["..."],"clarifying_questions":["..."]}'
        ),
        "search": (
            'OUTPUT FORMAT: Return valid JSON:\n'
            '{"search_strategy":"...","keywords":["..."],"sources":["..."],'
            '"refined_query":"...","next_steps":["..."]}'
        ),
        "codeReview": (
            'OUTPUT FORMAT: Return valid JSON:\n'
            '{"overall_rating":"excellent|good|needs_improvement|poor","summary":"...",'
            '"issues":[{"type":"bug|performance|security|style","severity":"critical|high|medium|low",'
            '"line":"...","description":"...","suggestion":"..."}],'
            '"strengths":["..."],"recommendations":["..."]}'
        ),
    }
    return formats.get(feature, formats["qna"])


# ── ChatEngine ─────────────────────────────────────────────────────────────

class ChatEngine:
    """
    Manages a single conversation session with Claude via AWS Bedrock.

    History is kept in two layers:
      conversation_history — loaded from DynamoDB on init (persisted turns)
      session_exchanges    — in-memory turns not yet persisted

    Both layers are concatenated into the prompt on every send_message() call.
    """

    def __init__(
        self,
        chat_id: str,
        system_prompt: Optional[str] = None,
        mode: ModeType = "api",
        use_dynamo: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.chat_id = chat_id
        self.system_prompt = system_prompt
        self.mode = mode
        self.use_dynamo = use_dynamo and DYNAMO_AVAILABLE
        self.metadata: Dict[str, Any] = metadata or {}

        if self.use_dynamo:
            self.dynamo_storage = DynamoChatStorage()
            self.conversation_history: List[Dict[str, str]] = (
                self.dynamo_storage.get_chat_history(chat_id)
            )
        else:
            self.conversation_history = []

        self.session_exchanges: List[Dict[str, str]] = []

    # ── Public API ─────────────────────────────────────────────────────────

    def send_message(self, user_input: str) -> str:
        """Send a message and return Claude's reply. Full history is included."""
        try:
            messages = self._build_messages(user_input)
            response = bedrock_client.generate(
                messages,
                options={"temperature": 0.0},
            )
            self.session_exchanges.append({"user": user_input, "assistant": response})
            return response
        except Exception as exc:
            error_msg = f"Error communicating with Claude: {exc}"
            logger.error(error_msg)
            return error_msg

    def save_to_dynamo(self) -> bool:
        """Flush session_exchanges to DynamoDB, then reload conversation_history."""
        if not self.use_dynamo or not self.session_exchanges:
            return False
        try:
            for exchange in self.session_exchanges:
                success = self.dynamo_storage.append_exchange(
                    self.chat_id, exchange["user"], exchange["assistant"], self.metadata
                )
                if not success:
                    return False
            self.session_exchanges.clear()
            self.conversation_history = self.dynamo_storage.get_chat_history(self.chat_id)
            return True
        except Exception as exc:
            logger.error("DynamoDB save failed: %s", exc)
            return False

    def get_complete_history(self) -> List[Dict[str, str]]:
        return self.conversation_history + self.session_exchanges

    def get_session_history(self) -> List[Dict[str, str]]:
        return list(self.session_exchanges)

    def get_dynamo_history(self) -> List[Dict[str, str]]:
        return list(self.conversation_history)

    def set_system_prompt(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt

    # ── Private helpers ────────────────────────────────────────────────────

    def _build_messages(self, user_input: str) -> List[Dict[str, str]]:
        """
        Build a message list for BedrockClient.generate().
        Uses the standard [{"role": ..., "content": ...}] format.
        """
        messages: List[Dict[str, str]] = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        for exchange in self.get_complete_history():
            messages.append({"role": "user", "content": exchange["user"]})
            messages.append({"role": "assistant", "content": exchange["assistant"]})

        messages.append({"role": "user", "content": user_input})
        return messages


# ── Interactive helpers ────────────────────────────────────────────────────

def create_interactive_chat(chat_id: Optional[str] = None) -> None:
    if chat_id is None:
        chat_id = (
            f"interactive_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        )
    print("Claude Chat (type 'quit' / 'exit' to stop)")
    print("-" * 50)
    chat = ChatEngine(
        chat_id=chat_id,
        system_prompt="You are a helpful AI assistant.",
        mode="chat",
        use_dynamo=False,
    )
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"quit", "exit", "bye"}:
                print("Goodbye!")
                break
            print("Claude:", chat.send_message(user_input))
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    create_interactive_chat()
