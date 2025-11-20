from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ChatRequest(BaseModel):
    message: str
    chat_id: Optional[str] = None
    system_prompt: Optional[str] = None
    feature: str = "qna"
    metadata: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    chat_id: str
    history: Optional[List[Dict[str, str]]] = None
