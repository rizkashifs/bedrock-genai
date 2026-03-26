"""
Tests for Settings/config, FileRegistry, and RefusalAgent.
"""
import os

import pytest

from app.config.settings import Settings
from app.services.registry import FileRegistry
from app.agents.refusal import RefusalAgent


# ── Settings ──────────────────────────────────────────────────────────────────

class TestSettings:
    def test_defaults_use_haiku(self):
        s = Settings()
        assert "haiku" in s.bedrock_model_id

    def test_model_type_haiku(self):
        s = Settings(bedrock_model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0")
        assert s.model_type == "haiku"

    def test_model_type_sonnet(self):
        s = Settings(bedrock_model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0")
        assert s.model_type == "sonnet"

    def test_max_tokens_haiku(self):
        s = Settings(bedrock_model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0")
        assert s.max_tokens == 4_096

    def test_max_tokens_sonnet(self):
        s = Settings(bedrock_model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0")
        assert s.max_tokens == 200_000

    def test_max_csv_rows_haiku(self):
        s = Settings(bedrock_model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0")
        assert s.max_csv_rows == 300

    def test_max_csv_rows_sonnet(self):
        s = Settings(bedrock_model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0")
        assert s.max_csv_rows is None

    def test_default_region(self):
        s = Settings()
        assert s.aws_region == "us-east-2"

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("AWS_REGION", "eu-west-1")
        s = Settings()
        assert s.aws_region == "eu-west-1"

    def test_titan_embed_model_id_default(self):
        s = Settings()
        assert s.titan_embed_model_id == "amazon.titan-embed-text-v2:0"

    def test_max_chunks_per_query_default(self):
        s = Settings()
        assert s.max_chunks_per_query == 20

    def test_base_dir_is_absolute(self):
        s = Settings()
        assert s.base_dir.is_absolute()


# ── FileRegistry ──────────────────────────────────────────────────────────────

@pytest.fixture
def registry():
    return FileRegistry()


class TestFileRegistry:
    def test_add_and_get_file(self, registry):
        registry.add_file(
            file_path="/data/sales.csv",
            summary="Monthly sales data",
            row_count=500,
            data_type="csv",
            schema_context="date, amount, region",
            semantic_summary="Sales records for 2023",
            text_heavy=False,
        )
        info = registry.get_file_info("/data/sales.csv")
        assert info is not None
        assert info["row_count"] == 500
        assert info["type"] == "csv"
        assert info["file_name"] == "sales.csv"

    def test_get_nonexistent_returns_none(self, registry):
        assert registry.get_file_info("/data/ghost.csv") is None

    def test_list_files_empty(self, registry):
        assert registry.list_files() == []

    def test_list_files_after_add(self, registry):
        registry.add_file("/a.csv", "A", 10, "csv", "", "", False)
        registry.add_file("/b.csv", "B", 20, "csv", "", "", True)
        paths = registry.list_files()
        assert "/a.csv" in paths
        assert "/b.csv" in paths

    def test_overwrite_existing_file(self, registry):
        registry.add_file("/data/f.csv", "Old summary", 100, "csv", "", "", False)
        registry.add_file("/data/f.csv", "New summary", 200, "csv", "", "", True)
        info = registry.get_file_info("/data/f.csv")
        assert info["summary"] == "New summary"
        assert info["row_count"] == 200

    def test_get_all_summaries_includes_all_files(self, registry):
        registry.add_file("/a.csv", "Summary A", 10, "csv", "", "", False)
        registry.add_file("/b.csv", "Summary B", 20, "csv", "", "", False)
        summaries = registry.get_all_summaries()
        assert "Summary A" in summaries
        assert "Summary B" in summaries

    def test_get_all_summaries_empty_registry(self, registry):
        assert registry.get_all_summaries() == ""

    def test_text_heavy_flag_stored(self, registry):
        registry.add_file("/heavy.csv", "Heavy text", 50, "csv", "", "", True)
        info = registry.get_file_info("/heavy.csv")
        assert info["text_heavy"] is True


# ── RefusalAgent ──────────────────────────────────────────────────────────────

class TestRefusalAgent:
    @pytest.fixture
    def agent(self):
        return RefusalAgent()

    def test_run_returns_relevant_rows_key(self, agent):
        result = agent.run({"schema_context": "id, name", "route_schema": {}})
        assert "relevant_rows" in result

    def test_run_contains_single_row(self, agent):
        result = agent.run({})
        assert len(result["relevant_rows"]) == 1

    def test_run_row_has_should_ask_user_true(self, agent):
        result = agent.run({})
        assert result["relevant_rows"][0]["should_ask_user"] is True

    def test_run_row_has_summary(self, agent):
        result = agent.run({})
        assert "_summary" in result["relevant_rows"][0]

    def test_build_summary_with_questions(self, agent):
        summary = agent._build_summary("id, name", ["Which column?", "What year?"])
        assert "Which column?" in summary
        assert "What year?" in summary

    def test_build_summary_without_questions(self, agent):
        summary = agent._build_summary("id, name, salary", [])
        assert "id, name, salary" in summary

    def test_follow_up_questions_capped_at_four(self, agent):
        questions = [f"Q{i}" for i in range(10)]
        summary = agent._build_summary("schema", questions)
        # Only first 4 should appear
        assert "Q0" in summary
        assert "Q3" in summary
        assert "Q4" not in summary

    def test_run_with_follow_up_questions(self, agent):
        result = agent.run({
            "schema_context": "col1, col2",
            "route_schema": {"follow_up_questions": ["What date range?"]},
        })
        assert "What date range?" in result["relevant_rows"][0]["_summary"]
