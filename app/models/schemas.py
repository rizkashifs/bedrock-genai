"""Pydantic request/response schemas for API endpoints."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


# ── Ingestion ──────────────────────────────────────────────────────────────

class IngestionRequest(BaseModel):
    s3_bucket: Optional[str] = None
    s3_key: Optional[str] = None
    file_path: Optional[str] = None   # local path (non-Lambda usage)
    user_prompt: str


class IngestionResponse(BaseModel):
    status: str
    answer: str
    file_summary: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


# ── Chat ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str
    chat_id: Optional[str] = None
    file_path: Optional[str] = None
    feature: str = "qna"   # qna | docComparison | search | codeReview
    mode: str = "chat"     # chat | api


class ChatResponse(BaseModel):
    answer: str
    chat_id: str
    question_type: Optional[str] = None
    retrieved_data: Optional[Any] = None
    file_summary: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


# ── Query (generic RAG) ────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    file_path: Optional[str] = None
    chat_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    contexts: List[Dict[str, Any]] = []
    metadata: Optional[Dict[str, Any]] = None
