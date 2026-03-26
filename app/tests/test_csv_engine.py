"""
Unit tests for SQLEngine (app/engines/csv_engine.py).

Tests cover:
- Column token normalisation
- Column resolution (exact, case-insensitive, fuzzy)
- _looks_like_year helper
- _get_filter_description
- _summary_rows_from_scalar output shapes
- _safe_float helper
- Supported operations set
"""
import pandas as pd
import pytest

from app.engines.csv_engine import SQLEngine


@pytest.fixture
def engine():
    return SQLEngine()


@pytest.fixture
def sales():
    return pd.DataFrame(
        {
            "Sale ID": [1, 2, 3, 4],
            "Amount": [1000.0, 2500.0, 750.0, 3200.0],
            "Region": ["North", "South", "North", "East"],
            "Sale Date": ["2023-01-01", "2023-02-15", "2023-03-10", "2023-04-15"],
        }
    )


# ── _normalize_column_token ───────────────────────────────────────────────────

class TestNormalizeColumnToken:
    def test_lowercases_and_strips(self, engine):
        assert engine._normalize_column_token("  Hello  ") == "hello"

    def test_removes_non_alphanumeric(self, engine):
        assert engine._normalize_column_token("Sale-ID_2023!") == "saleid2023"

    def test_empty(self, engine):
        assert engine._normalize_column_token("") == ""

    def test_numbers_preserved(self, engine):
        assert engine._normalize_column_token("Col99") == "col99"


# ── _resolve_column ───────────────────────────────────────────────────────────

class TestResolveColumn:
    def test_exact_match(self, engine, sales):
        assert engine._resolve_column(sales, "Amount") == "Amount"

    def test_case_insensitive(self, engine, sales):
        assert engine._resolve_column(sales, "amount") == "Amount"
        assert engine._resolve_column(sales, "REGION") == "Region"

    def test_fuzzy_token_match(self, engine, sales):
        # "Sale ID" → normalised "saleid"; "Sale_ID" → "saleid"
        assert engine._resolve_column(sales, "Sale_ID") == "Sale ID"

    def test_returns_none_for_unknown(self, engine, sales):
        assert engine._resolve_column(sales, "nonexistent") is None

    def test_returns_none_for_empty(self, engine, sales):
        assert engine._resolve_column(sales, "") is None

    def test_returns_none_for_none(self, engine, sales):
        assert engine._resolve_column(sales, None) is None


# ── _resolve_columns ──────────────────────────────────────────────────────────

class TestResolveColumns:
    def test_resolves_multiple(self, engine, sales):
        result = engine._resolve_columns(sales, ["amount", "region"])
        assert result == ["Amount", "Region"]

    def test_deduplication(self, engine, sales):
        result = engine._resolve_columns(sales, ["Amount", "Amount"])
        assert result == ["Amount"]

    def test_skips_unknown_column(self, engine, sales):
        result = engine._resolve_columns(sales, ["amount", "ghost"])
        assert result == ["Amount"]

    def test_empty_input(self, engine, sales):
        assert engine._resolve_columns(sales, []) == []

    def test_none_input(self, engine, sales):
        assert engine._resolve_columns(sales, None) == []


# ── _looks_like_year ──────────────────────────────────────────────────────────

class TestLooksLikeYear:
    def test_four_digit_number(self, engine):
        assert engine._looks_like_year("2023") is True
        assert engine._looks_like_year(2023) is True

    def test_non_four_digit(self, engine):
        assert engine._looks_like_year("20230") is False
        assert engine._looks_like_year("23") is False
        assert engine._looks_like_year("abcd") is False

    def test_decimal_not_year(self, engine):
        assert engine._looks_like_year("2023.0") is False


# ── _get_filter_description ───────────────────────────────────────────────────

class TestGetFilterDescription:
    def test_empty_filters_returns_empty(self, engine):
        assert engine._get_filter_description([]) == ""
        assert engine._get_filter_description(None) == ""

    def test_single_equals_filter(self, engine):
        filters = [{"column": "Region", "operator": "=", "value": "North"}]
        desc = engine._get_filter_description(filters)
        assert "where" in desc.lower()
        assert "Region" in desc
        assert "North" in desc

    def test_date_column_uses_date_language(self, engine):
        filters = [{"column": "sale_date", "operator": ">=", "value": "2023-01-01"}]
        desc = engine._get_filter_description(filters)
        assert "after or on" in desc

    def test_multiple_filters_joined(self, engine):
        filters = [
            {"column": "Region", "operator": "=", "value": "North"},
            {"column": "Amount", "operator": ">", "value": 1000},
        ]
        desc = engine._get_filter_description(filters)
        assert "and" in desc


# ── _summary_rows_from_scalar ─────────────────────────────────────────────────

class TestSummaryRowsFromScalar:
    def test_count_operation(self, engine):
        rows = engine._summary_rows_from_scalar("count", 42, ["id"])
        assert len(rows) == 1
        assert "42" in rows[0]["_summary"]

    def test_dict_result_produces_one_row_per_column(self, engine):
        result = {"salary": 75000.0, "bonus": 5000.0}
        rows = engine._summary_rows_from_scalar("avg", result, ["salary", "bonus"])
        assert len(rows) == 2
        col_keys = [list(r.keys())[0] for r in rows]
        assert "salary" in col_keys
        assert "bonus" in col_keys

    def test_error_reason_included_in_fallback(self, engine):
        rows = engine._summary_rows_from_scalar(
            "sum", None, ["price"], error_reason="column is non-numeric"
        )
        assert rows[0].get("should_ask_user") is True
        assert "non-numeric" in rows[0]["_summary"]

    def test_histogram_list_passthrough(self, engine):
        histogram = [{"bucket": "0-100", "count": 5, "_summary": "5 rows"}]
        rows = engine._summary_rows_from_scalar("histogram", histogram, ["price"])
        assert rows == histogram

    def test_correlation_dict_passthrough(self, engine):
        corr = {"price": {"price": 1.0, "stock": -0.3}}
        rows = engine._summary_rows_from_scalar("correlation", corr, ["price", "stock"])
        assert rows[0]["correlation"] == corr


# ── _safe_float ───────────────────────────────────────────────────────────────

class TestSafeFloat:
    def test_converts_int(self, engine):
        assert engine._safe_float(5) == 5.0

    def test_converts_string_number(self, engine):
        assert engine._safe_float("3.14") == pytest.approx(3.14)

    def test_returns_none_for_nan(self, engine):
        import math
        assert engine._safe_float(float("nan")) is None

    def test_returns_value_for_unconvertible(self, engine):
        assert engine._safe_float("not_a_number") == "not_a_number"


# ── Supported operations set ──────────────────────────────────────────────────

class TestSupportedOperations:
    def test_all_expected_operations_present(self):
        expected = {
            "sum", "avg", "count", "max", "min", "median", "mode",
            "std", "variance", "quantile", "histogram", "value_counts",
            "distinct_count", "null_count", "null_pct", "correlation",
            "profile", "filter", "none",
        }
        assert expected == SQLEngine._SUPPORTED_OPERATIONS

    def test_operation_aliases_map_to_supported(self):
        for alias, canonical in SQLEngine._OPERATION_ALIASES.items():
            assert canonical in SQLEngine._SUPPORTED_OPERATIONS, (
                f"Alias '{alias}' maps to '{canonical}' which is not in _SUPPORTED_OPERATIONS"
            )
