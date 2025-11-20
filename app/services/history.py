from typing import List, Dict, Optional

# Simple in-memory storage for demonstration purposes
# In a real app, this would be a database (DynamoDB, Postgres, etc.)
_HISTORY_STORE: Dict[str, List[Dict[str, str]]] = {}

class HistoryService:
    def get_history(self, chat_id: str) -> List[Dict[str, str]]:
        """Retrieve conversation history for a given chat_id."""
        return _HISTORY_STORE.get(chat_id, [])

    def add_exchange(self, chat_id: str, user_msg: str, assistant_msg: str):
        """Add a user-assistant exchange to the history."""
        if chat_id not in _HISTORY_STORE:
            _HISTORY_STORE[chat_id] = []
        
        _HISTORY_STORE[chat_id].append({"user": user_msg, "assistant": assistant_msg})

    def clear_history(self, chat_id: str):
        """Clear history for a chat_id."""
        if chat_id in _HISTORY_STORE:
            del _HISTORY_STORE[chat_id]
