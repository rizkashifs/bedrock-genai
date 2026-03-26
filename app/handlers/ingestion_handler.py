"""
Lambda-compatible document ingestion handler.

Accepts either:
  - S3-based event: {"s3_bucket": "...", "s3_key": "path/doc.pdf", "user_prompt": "..."}
  - Local-path event: {"file_path": "/abs/path/doc.csv", "user_prompt": "..."}

For PDF files, chunks are indexed into the global RetrievalService so they
can be queried immediately after ingestion.

For CSV/Excel files, the file is registered in the FileRegistry for use by
the OrchestrationService.
"""

from typing import Any, Dict

from app.services.document_processing import process_file
from app.services.registry import file_registry
from app.services.retrieval import get_global_retrieval_service
from app.services.orchestration import orchestrator
from app.utils.logger import logger


def handle_ingestion(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main ingestion entry point (Lambda handler or direct call).

    Returns a response dict with statusCode and body.
    """
    bucket = event.get("s3_bucket")
    key = event.get("s3_key")
    file_path: str = event.get("file_path", "")
    user_prompt: str = event.get("user_prompt", "")

    # ── Resolve file path ──────────────────────────────────────────────────
    if not file_path:
        if not bucket or not key:
            return {
                "statusCode": 400,
                "body": {"error": "Provide either 'file_path' (local) or both 's3_bucket' and 's3_key'."},
            }
        # TODO: download from S3 to a temp path and set file_path
        return {
            "statusCode": 501,
            "body": {"error": "S3 download is not yet implemented. Pass 'file_path' for local files."},
        }

    if not user_prompt:
        return {
            "statusCode": 400,
            "body": {"error": "Missing required parameter: user_prompt"},
        }

    logger.info("Ingestion requested. file=%s", file_path)

    # ── Process the file ───────────────────────────────────────────────────
    try:
        result = process_file(file_path, user_question=user_prompt)
    except Exception as exc:
        logger.error("process_file failed: %s", exc)
        return {"statusCode": 500, "body": {"error": str(exc)}}

    if result["status"] == "error":
        return {
            "statusCode": 422,
            "body": {"error": result["message"]},
        }

    file_type = result["type"]

    # ── Post-processing per type ───────────────────────────────────────────
    if file_type == "pdf":
        chunks = result.get("chunks", [])
        retrieval_svc = get_global_retrieval_service()
        retrieval_svc.index_chunks(chunks)
        logger.info("Indexed %d chunks into retrieval service", len(chunks))
        answer = (
            f"Processed '{file_path}': {len(chunks)} chunks indexed. "
            f"User asked: '{user_prompt}'."
        )
        return {
            "statusCode": 200,
            "body": {
                "answer": answer,
                "stats": result.get("stats", {}),
                "chunks_indexed": len(chunks),
            },
        }

    elif file_type == "csv":
        # Register the file for CSV/Excel orchestration queries
        data_bundle = result.get("data_bundle", {})
        try:
            orchestrator.register_data(file_path, data_bundle)
            logger.info("Registered CSV/Excel in FileRegistry: %s", file_path)
        except Exception as exc:
            logger.error("FileRegistry registration failed: %s", exc)
        answer = (
            f"Processed '{file_path}': {data_bundle.get('row_count', '?')} rows loaded. "
            f"User asked: '{user_prompt}'."
        )
        return {
            "statusCode": 200,
            "body": {
                "answer": answer,
                "row_count": data_bundle.get("row_count"),
                "type": data_bundle.get("type"),
            },
        }

    elif file_type == "image":
        return {
            "statusCode": 200,
            "body": {"answer": result.get("answer", "")},
        }

    return {
        "statusCode": 200,
        "body": {"answer": result.get("message", "File processed.")},
    }
