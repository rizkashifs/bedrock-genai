"""
Unit tests for TextEngine (app/engines/text_engine.py).

Tests cover:
- Column name resolution (exact, case-insensitive, fuzzy)
- Series normalisation
- Keyword extraction
- ID filter application
- Post-filter operators (=, !=, contains, in, >, <, >=, <=)
- Full execute() pipeline
"""
import pandas as pd
import pytest

from app.engines.text_engine import TextEngine


@pytest.fixture
def engine():
    return TextEngine()


@pytest.fixture
def employees():
    return pd.DataFrame(
        {
            "employee_id": [1001, 1002, 1003, 1004, 1005],
            "Full Name": ["Alice Smith", "Bob Jones", "Carol Lee", "Dave Brown", "Eve White"],
            "Department": ["Engineering", "Marketing", "Engineering", "HR", "Marketing"],
            "Salary": [90000, 75000, 95000, 60000, 80000],
            "Notes": [
                "Strong Python skills",
                "Leads social media campaigns",
                "Senior engineer great mentor",
                "Handles payroll and benefits",
                "Digital marketing specialist",
            ],
        }
    )


# ── _normalize_token ─────────────────────────────────────────────────────────

class TestNormalizeToken:
    def test_lowercase_stripping(self, engine):
        assert engine._normalize_token("  Hello World  ") == "helloworld"

    def test_removes_special_chars(self, engine):
        assert engine._normalize_token("Unit-Test_Col!") == "unittestcol"

    def test_numeric_preserved(self, engine):
        assert engine._normalize_token("Col123") == "col123"

    def test_empty_string(self, engine):
        assert engine._normalize_token("") == ""


# ── _resolve_column ──────────────────────────────────────────────────────────

class TestResolveColumn:
    def test_exact_match(self, engine, employees):
        assert engine._resolve_column(employees, "Department") == "Department"

    def test_case_insensitive_match(self, engine, employees):
        assert engine._resolve_column(employees, "department") == "Department"
        assert engine._resolve_column(employees, "SALARY") == "Salary"

    def test_fuzzy_token_match(self, engine, employees):
        # "Full Name" normalised → "fullname"; "Full_Name" → "fullname"
        assert engine._resolve_column(employees, "Full_Name") == "Full Name"

    def test_no_match_returns_none(self, engine, employees):
        assert engine._resolve_column(employees, "nonexistent_col") is None

    def test_none_input_returns_none(self, engine, employees):
        assert engine._resolve_column(employees, None) is None


# ── _resolve_columns ─────────────────────────────────────────────────────────

class TestResolveColumns:
    def test_resolves_multiple(self, engine, employees):
        result = engine._resolve_columns(employees, ["department", "salary"])
        assert result == ["Department", "Salary"]

    def test_deduplicates(self, engine, employees):
        result = engine._resolve_columns(employees, ["Department", "Department"])
        assert result == ["Department"]

    def test_skips_unresolvable(self, engine, employees):
        result = engine._resolve_columns(employees, ["department", "ghost_col"])
        assert result == ["Department"]

    def test_empty_list(self, engine, employees):
        assert engine._resolve_columns(employees, []) == []

    def test_none_list(self, engine, employees):
        assert engine._resolve_columns(employees, None) == []


# ── _normalise_series ────────────────────────────────────────────────────────

class TestNormaliseSeries:
    def test_lowercases(self):
        s = pd.Series(["HELLO", "World"])
        result = TextEngine._normalise_series(s)
        assert list(result) == ["hello", "world"]

    def test_strips_whitespace(self):
        s = pd.Series(["  hello  ", " world "])
        result = TextEngine._normalise_series(s)
        assert list(result) == ["hello", "world"]

    def test_collapses_internal_whitespace(self):
        s = pd.Series(["hello   world"])
        result = TextEngine._normalise_series(s)
        assert result.iloc[0] == "hello world"

    def test_fills_nan(self):
        s = pd.Series(["hello", None, float("nan")])
        result = TextEngine._normalise_series(s)
        assert result.iloc[1] == ""
        assert result.iloc[2] == ""


# ── _extract_keywords ────────────────────────────────────────────────────────

class TestExtractKeywords:
    def test_explicit_keywords_returned_as_is(self):
        plan = {"keywords": ["Python", "Engineer", "Senior"]}
        result = TextEngine._extract_keywords(plan)
        assert result == ["python", "engineer", "senior"]

    def test_falls_back_to_query_text(self):
        plan = {"query_text": "Find senior Python engineers"}
        result = TextEngine._extract_keywords(plan)
        assert "python" in result
        assert "engineers" in result
        # stopwords removed
        assert "find" not in result

    def test_removes_short_tokens(self):
        plan = {"query_text": "is it an awesome day"}
        result = TextEngine._extract_keywords(plan)
        # "is", "it", "an", "day" — short words / stopwords removed
        # "awesome" should survive
        assert "awesome" in result

    def test_empty_plan_returns_empty(self):
        assert TextEngine._extract_keywords({}) == []

    def test_explicit_keywords_take_precedence(self):
        plan = {"keywords": ["alpha"], "query_text": "something completely different"}
        result = TextEngine._extract_keywords(plan)
        assert result == ["alpha"]


# ── _apply_id_filters ────────────────────────────────────────────────────────

class TestApplyIdFilters:
    def test_numeric_equality_filter(self, engine, employees):
        issues = []
        filtered = engine._apply_id_filters(
            employees, [{"column": "employee_id", "value": "1002"}], issues
        )
        assert len(filtered) == 1
        assert filtered.iloc[0]["Full Name"] == "Bob Jones"
        assert not issues

    def test_string_equality_filter(self, engine, employees):
        issues = []
        filtered = engine._apply_id_filters(
            employees, [{"column": "Department", "value": "Engineering"}], issues
        )
        assert len(filtered) == 2

    def test_case_insensitive_string_filter(self, engine, employees):
        issues = []
        filtered = engine._apply_id_filters(
            employees, [{"column": "department", "value": "MARKETING"}], issues
        )
        assert len(filtered) == 2

    def test_missing_column_adds_issue(self, engine, employees):
        issues = []
        result = engine._apply_id_filters(
            employees, [{"column": "ghost_column", "value": "x"}], issues
        )
        assert len(issues) == 1
        assert "ghost_column" in issues[0]
        assert len(result) == len(employees)  # unchanged

    def test_empty_filters_no_op(self, engine, employees):
        issues = []
        result = engine._apply_id_filters(employees, [], issues)
        assert len(result) == len(employees)


# ── _apply_post_filter ───────────────────────────────────────────────────────

class TestApplyPostFilter:
    def test_equals_numeric(self, engine, employees):
        issues = []
        result = engine._apply_post_filter(
            employees, {"column": "Salary", "operator": "=", "value": 75000}, issues
        )
        assert len(result) == 1
        assert result.iloc[0]["Full Name"] == "Bob Jones"

    def test_not_equals(self, engine, employees):
        issues = []
        result = engine._apply_post_filter(
            employees, {"column": "Department", "operator": "!=", "value": "Engineering"}, issues
        )
        assert len(result) == 3
        assert all(d != "Engineering" for d in result["Department"])

    def test_contains(self, engine, employees):
        issues = []
        result = engine._apply_post_filter(
            employees, {"column": "Notes", "operator": "contains", "value": "engineer"}, issues
        )
        assert len(result) == 1  # Carol only ("Senior engineer great mentor")

    def test_greater_than(self, engine, employees):
        issues = []
        result = engine._apply_post_filter(
            employees, {"column": "Salary", "operator": ">", "value": 80000}, issues
        )
        assert all(s > 80000 for s in result["Salary"])

    def test_less_than_equal(self, engine, employees):
        issues = []
        result = engine._apply_post_filter(
            employees, {"column": "Salary", "operator": "<=", "value": 75000}, issues
        )
        assert all(s <= 75000 for s in result["Salary"])

    def test_in_operator(self, engine, employees):
        issues = []
        result = engine._apply_post_filter(
            employees,
            {"column": "Department", "operator": "in", "value": ["Engineering", "HR"]},
            issues,
        )
        assert set(result["Department"].unique()) == {"Engineering", "HR"}

    def test_not_in_operator(self, engine, employees):
        issues = []
        result = engine._apply_post_filter(
            employees,
            {"column": "Department", "operator": "not in", "value": ["Engineering"]},
            issues,
        )
        assert "Engineering" not in result["Department"].values

    def test_missing_column_adds_issue(self, engine, employees):
        issues = []
        result = engine._apply_post_filter(
            employees, {"column": "ghost", "operator": "=", "value": "x"}, issues
        )
        assert len(issues) == 1
        assert len(result) == len(employees)


# ── execute() full pipeline ───────────────────────────────────────────────────

class TestExecute:
    def _intent(self, query_text="", keywords=None, target_cols=None,
                 id_filters=None, post_filters=None, top_k=10):
        return {
            "semantic_plan": {
                "query_text": query_text,
                "keywords": keywords or [],
                "target_text_columns": target_cols or [],
                "id_filters": id_filters or [],
                "post_filters": post_filters or [],
                "top_k": top_k,
            }
        }

    def test_keyword_match_returns_relevant_rows(self, engine, employees):
        intent = self._intent(keywords=["python"])
        result = engine.execute(employees, intent)
        assert "relevant_rows" in result
        assert len(result["relevant_rows"]) >= 1
        assert any("Python" in r.get("Notes", "") for r in result["relevant_rows"])

    def test_keyword_match_case_insensitive(self, engine, employees):
        intent = self._intent(keywords=["MARKETING"])
        result = engine.execute(employees, intent)
        rows = result["relevant_rows"]
        assert len(rows) >= 1

    def test_and_logic_strict_then_or_fallback(self, engine, employees):
        # Both keywords must not all co-occur → should fall back to OR
        intent = self._intent(keywords=["python", "marketing"])
        result = engine.execute(employees, intent)
        # At least one result (OR fallback)
        assert "relevant_rows" in result
        assert len(result["relevant_rows"]) >= 1

    def test_id_filter_scopes_result(self, engine, employees):
        # Alice (1001) has "Strong Python skills" — search for "python" after scoping
        intent = self._intent(
            keywords=["python"],
            id_filters=[{"column": "employee_id", "value": "1001"}],
        )
        result = engine.execute(employees, intent)
        rows = result["relevant_rows"]
        assert len(rows) == 1
        assert rows[0]["employee_id"] == 1001

    def test_post_filter_applied_after_text_search(self, engine, employees):
        # "engineer" matches Carol (95000) only; post_filter salary > 90000 keeps her
        intent = self._intent(
            keywords=["engineer"],
            post_filters=[{"column": "Salary", "operator": ">", "value": 90000}],
        )
        result = engine.execute(employees, intent)
        assert "relevant_rows" in result
        rows = result["relevant_rows"]
        # All matched rows must pass the salary filter
        data_rows = [r for r in rows if not r.get("should_ask_user")]
        assert len(data_rows) >= 1
        assert all(r["Salary"] > 90000 for r in data_rows)

    def test_top_k_respected(self, engine, employees):
        intent = self._intent(keywords=["a"], top_k=2)
        result = engine.execute(employees, intent)
        assert len(result.get("relevant_rows", [])) <= 2

    def test_no_match_returns_refusal(self, engine, employees):
        intent = self._intent(keywords=["zzz_no_match_zzz"])
        result = engine.execute(employees, intent)
        assert "relevant_rows" in result
        rows = result["relevant_rows"]
        assert len(rows) == 1
        assert rows[0].get("should_ask_user") is True

    def test_empty_query_and_no_id_filters_returns_refusal(self, engine, employees):
        intent = self._intent(query_text="", keywords=[])
        result = engine.execute(employees, intent)
        rows = result["relevant_rows"]
        assert rows[0].get("should_ask_user") is True

    def test_summary_key_present_in_results(self, engine, employees):
        intent = self._intent(keywords=["marketing"])
        result = engine.execute(employees, intent)
        for row in result["relevant_rows"]:
            assert "_summary" in row

    def test_multi_dataframe_input(self, engine, employees):
        df2 = pd.DataFrame(
            {"project": ["Alpha", "Beta"], "lead": ["Alice", "Carol"], "budget": [50000, 70000]}
        )
        intent = self._intent(keywords=["alpha"])
        result = engine.execute({"employees": employees, "projects": df2}, intent)
        assert "relevant_rows" in result

    def test_target_columns_respected(self, engine, employees):
        # Only search in "Notes"; "Python" is in Notes only for Alice
        intent = self._intent(keywords=["python"], target_cols=["Notes"])
        result = engine.execute(employees, intent)
        rows = result["relevant_rows"]
        assert any("Python" in r.get("Notes", "") for r in rows)
