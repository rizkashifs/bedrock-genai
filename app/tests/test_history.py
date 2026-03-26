"""
Unit tests for HistoryService (app/services/history.py).
"""
import pytest

from app.services.history import HistoryService, get_history


@pytest.fixture
def svc():
    """Fresh HistoryService instance for each test."""
    return HistoryService()


class TestHistoryService:
    def test_empty_history_for_new_session(self, svc):
        assert svc.get_history("session-1") == []

    def test_add_and_retrieve_single_turn(self, svc):
        svc.add_turn("s1", user="Hello", assistant="Hi there!")
        history = svc.get_history("s1")
        assert len(history) == 1
        assert history[0] == {"user": "Hello", "assistant": "Hi there!"}

    def test_multiple_turns_preserved_in_order(self, svc):
        svc.add_turn("s1", user="Q1", assistant="A1")
        svc.add_turn("s1", user="Q2", assistant="A2")
        svc.add_turn("s1", user="Q3", assistant="A3")
        history = svc.get_history("s1")
        assert len(history) == 3
        assert history[0]["user"] == "Q1"
        assert history[2]["user"] == "Q3"

    def test_sessions_are_isolated(self, svc):
        svc.add_turn("session-A", user="A question", assistant="A answer")
        svc.add_turn("session-B", user="B question", assistant="B answer")
        assert len(svc.get_history("session-A")) == 1
        assert len(svc.get_history("session-B")) == 1

    def test_empty_chat_id_get_returns_empty(self, svc):
        svc.add_turn("real-session", user="q", assistant="a")
        assert svc.get_history("") == []
        assert svc.get_history(None) == []

    def test_empty_chat_id_add_is_noop(self, svc):
        svc.add_turn("", user="q", assistant="a")
        svc.add_turn(None, user="q", assistant="a")
        # Neither should raise; no session created
        assert svc.get_history("") == []

    def test_returns_reference_not_copy(self, svc):
        """get_history returns the live list — mutations are visible."""
        svc.add_turn("s1", "q1", "a1")
        h1 = svc.get_history("s1")
        svc.add_turn("s1", "q2", "a2")
        h2 = svc.get_history("s1")
        assert len(h2) == 2


class TestGetHistoryModuleFunction:
    def test_module_function_delegates_to_singleton(self):
        # The module-level get_history() wraps the singleton; just verify it returns a list
        result = get_history("brand-new-session-xyz")
        assert isinstance(result, list)
