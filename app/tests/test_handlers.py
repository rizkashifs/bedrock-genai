"""
Unit tests for Lambda handlers.

Uses pytest-mock to isolate all external dependencies (Bedrock, file I/O).
"""
from unittest.mock import MagicMock, patch

import pytest

from app.handlers.ingestion_handler import handle_ingestion
from app.handlers.chat_handler import handle_chat


# ── handle_ingestion ──────────────────────────────────────────────────────────

class TestHandleIngestion:
    def test_missing_file_path_returns_400(self):
        result = handle_ingestion({})
        assert result["statusCode"] == 400
        import json
        body = json.loads(result["body"]) if isinstance(result["body"], str) else result["body"]
        assert "error" in body or "message" in body

    def test_missing_user_prompt_returns_400(self):
        result = handle_ingestion({"file_path": "/tmp/doc.pdf"})
        assert result["statusCode"] == 400

    @patch("app.handlers.ingestion_handler.process_file")
    def test_pdf_ingestion_indexes_chunks(self, mock_process_file):
        """PDF branch should call index_chunks on the retrieval service."""
        mock_chunk = MagicMock()
        mock_chunk.chunk_type = "text"
        mock_process_file.return_value = {
            "type": "pdf",
            "status": "success",
            "chunks": [mock_chunk],
        }

        with patch("app.handlers.ingestion_handler.get_global_retrieval_service") as mock_rs:
            mock_svc = MagicMock()
            mock_rs.return_value = mock_svc

            result = handle_ingestion({
                "file_path": "/tmp/test.pdf",
                "user_prompt": "Summarise this document",
            })

        assert result["statusCode"] == 200
        mock_svc.index_chunks.assert_called_once()

    @patch("app.handlers.ingestion_handler.process_file")
    def test_csv_ingestion_registers_with_orchestrator(self, mock_process_file):
        """CSV branch should register the data bundle with the orchestrator."""
        import pandas as pd
        mock_process_file.return_value = {
            "type": "csv",
            "status": "success",
            "data_bundle": {
                "df": pd.DataFrame({"a": [1, 2]}),
                "type": "csv",
                "semantic_context": "schema",
                "row_count": 2,
            },
        }

        with patch("app.handlers.ingestion_handler.orchestrator") as mock_orch:
            result = handle_ingestion({
                "file_path": "/data/sales.csv",
                "user_prompt": "Load this file",
            })

        assert result["statusCode"] == 200
        mock_orch.register_data.assert_called_once()

    @patch("app.handlers.ingestion_handler.process_file")
    def test_process_file_error_returns_500(self, mock_process_file):
        mock_process_file.side_effect = RuntimeError("Unexpected failure")

        result = handle_ingestion({
            "file_path": "/tmp/bad.pdf",
            "user_prompt": "Summarise",
        })

        assert result["statusCode"] == 500

    @patch("app.handlers.ingestion_handler.process_file")
    def test_failed_status_returns_error(self, mock_process_file):
        mock_process_file.return_value = {
            "type": "pdf",
            "status": "error",
            "message": "Could not parse PDF",
        }

        result = handle_ingestion({
            "file_path": "/tmp/bad.pdf",
            "user_prompt": "Summarise",
        })

        assert result["statusCode"] in (400, 422, 500)


# ── handle_chat ───────────────────────────────────────────────────────────────

class TestHandleChat:
    def test_missing_query_returns_400(self):
        result = handle_chat({})
        assert result["statusCode"] == 400

    @patch("app.handlers.chat_handler.orchestrator")
    def test_csv_route_uses_orchestrator(self, mock_orch):
        mock_orch.run_pipeline.return_value = {
            "answer": "Total sales: $10,000",
            "route": "SQL_ENGINE",
        }

        result = handle_chat({
            "query": "What are total sales?",
            "file_path": "/data/sales.csv",
        })

        assert result["statusCode"] == 200
        mock_orch.run_pipeline.assert_called_once()

    @patch("app.handlers.chat_handler.get_global_retrieval_service")
    @patch("app.handlers.chat_handler.bedrock_client")
    def test_pdf_route_uses_retrieval_service(self, mock_bedrock, mock_rs_fn):
        mock_svc = MagicMock()
        mock_retrieved = MagicMock()
        mock_retrieved.chunk.chunk_type = "text"
        mock_retrieved.chunk.text = "The document says X"
        mock_retrieved.chunk.metadata = {}
        mock_svc.retrieve_with_images.return_value = [mock_retrieved]
        mock_rs_fn.return_value = mock_svc
        mock_bedrock.generate.return_value = "Based on the document, X is the answer."

        result = handle_chat({
            "query": "What does the document say?",
            "file_path": "/data/report.pdf",
        })

        assert result["statusCode"] == 200
        mock_svc.retrieve_with_images.assert_called_once()

    @patch("app.handlers.chat_handler.ChatEngine")
    def test_general_route_uses_chat_engine(self, MockChatEngine):
        mock_engine = MagicMock()
        mock_engine.send_message.return_value = "A general answer."
        MockChatEngine.return_value = mock_engine

        result = handle_chat({
            "query": "What is the capital of France?",
        })

        assert result["statusCode"] == 200
        mock_engine.send_message.assert_called_once()

    @patch("app.handlers.chat_handler.orchestrator")
    def test_chat_id_returned_in_response(self, mock_orch):
        mock_orch.run_pipeline.return_value = {"answer": "42", "route": "SQL_ENGINE"}

        result = handle_chat({
            "query": "count rows",
            "file_path": "/data/data.xlsx",
            "chat_id": "session-abc",
        })

        import json
        body = json.loads(result["body"]) if isinstance(result["body"], str) else result["body"]
        assert body.get("chat_id") == "session-abc"

    @patch("app.handlers.chat_handler.ChatEngine")
    def test_exception_returns_500(self, MockChatEngine):
        MockChatEngine.side_effect = RuntimeError("Engine exploded")

        result = handle_chat({"query": "hello"})
        assert result["statusCode"] == 500
