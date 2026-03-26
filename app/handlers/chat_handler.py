"""
Chat request handler — routes queries to the correct pipeline.

PDF queries  -> RetrievalService + BedrockClient
CSV queries  -> OrchestrationService (SQL/Text engine + agents)
General chat -> ChatEngine (multi-turn conversation)
"""

import uuid
from typing import Any, Dict, List, Optional

from app.models.bedrock_client import bedrock_client
from app.services.chat_engine import ChatEngine, get_system_prompt
from app.services.history import history_service
from app.services.orchestration import orchestrator
from app.services.retrieval import get_global_retrieval_service
from app.utils.logger import logger


def handle_chat(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unified chat entry point.

    Event keys:
        query      (str, required)
        chat_id    (str, optional)  -- created if absent
        file_path  (str, optional)  -- routes to file-specific pipeline
        feature    (str, optional)  -- qna | docComparison | search | codeReview
        mode       (str, optional)  -- chat | api
    """
    query: str = event.get("query", "")
    if not query:
        return {"statusCode": 400, "body": {"error": "Missing required field: query"}}

    chat_id: str = event.get("chat_id") or f"chat_{str(uuid.uuid4())[:8]}"
    file_path: Optional[str] = event.get("file_path")
    feature: str = event.get("feature", "qna")
    mode: str = event.get("mode", "chat")

    logger.info("ChatHandler. chat_id=%s query=%s", chat_id, query[:60])

    # Route 1: CSV / Excel via SQL+Text engines
    if file_path and _is_tabular(file_path):
        try:
            result = orchestrator.run_pipeline(
                query=query,
                file_path=file_path,
                index_name=file_path.split("/")[-1],
                chat_id=chat_id,
            )
            return {
                "statusCode": 200,
                "body": {
                    "answer": result.get("answer", ""),
                    "chat_id": chat_id,
                    "question_type": result.get("question_type", "csv"),
                    "retrieved_data": result.get("retrieved_data"),
                    "file_summary": result.get("file_summary"),
                    "metadata": result.get("metadata", {}),
                },
            }
        except Exception as exc:
            logger.error("CSV pipeline error: %s", exc)
            return {"statusCode": 500, "body": {"error": str(exc)}}

    # Route 2: PDF via indexed chunks + Titan retrieval
    if file_path and _is_pdf(file_path):
        retrieval_svc = get_global_retrieval_service()
        history = history_service.get_history(chat_id)
        answer = _answer_from_pdf(query, retrieval_svc, history)
        history_service.add_turn(chat_id, query, answer)
        return {
            "statusCode": 200,
            "body": {"answer": answer, "chat_id": chat_id, "question_type": "pdf"},
        }

    # Route 3: Plain multi-turn conversation
    try:
        system_prompt = get_system_prompt(feature, mode=mode)  # type: ignore[arg-type]
        engine = ChatEngine(
            chat_id=chat_id,
            system_prompt=system_prompt,
            mode=mode,  # type: ignore[arg-type]
            use_dynamo=False,
        )
        for turn in history_service.get_history(chat_id):
            engine.conversation_history.append(turn)

        answer = engine.send_message(query)
        history_service.add_turn(chat_id, query, answer)

        return {
            "statusCode": 200,
            "body": {"answer": answer, "chat_id": chat_id, "question_type": "general"},
        }
    except Exception as exc:
        logger.error("Chat engine error: %s", exc)
        return {"statusCode": 500, "body": {"error": str(exc)}}


# ── Private helpers ────────────────────────────────────────────────────────

def _is_tabular(path: str) -> bool:
    return path.lower().endswith((".csv", ".xlsx", ".xls"))


def _is_pdf(path: str) -> bool:
    return path.lower().endswith(".pdf")


def _answer_from_pdf(
    query: str,
    retrieval_svc,
    history: List[Dict[str, str]],
) -> str:
    retrieved = retrieval_svc.retrieve_with_images(query)

    context_blocks: List[str] = []
    for r in retrieved:
        chunk = r.chunk
        if chunk.chunk_type == "text":
            context_blocks.append(
                f"[Doc {chunk.source_document_id} | page {chunk.page + 1}] {chunk.text}"
            )
            linked = chunk.metadata.get("linked_image_ids", [])
            if linked:
                context_blocks.append(f"  (section contains {len(linked)} image(s))")
        elif chunk.chunk_type == "image" and chunk.text:
            context_blocks.append(
                f"[Image | Doc {chunk.source_document_id} | page {chunk.page + 1}]\n"
                f"OCR: {chunk.text}"
            )

    context_text = "\n\n".join(context_blocks) or "No relevant context found."

    system_prompt = (
        "You are a PDF analysis assistant. Use ONLY the provided context from PDF chunks. "
        "Cite pages explicitly (e.g. 'See page 3'). If the answer is not in the context, say so."
    )
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for turn in history[-5:]:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    messages.append(
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
    )

    return bedrock_client.generate(messages, options={"temperature": 0.2, "max_tokens": 2048})
