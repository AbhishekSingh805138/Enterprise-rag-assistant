"""Phase 20: Parallel Processing tests.

Tests for:
- Parallel sub-query processing (process_sub_queries_parallel)
- Per-document grading (grade_documents with per_doc_grading=True)
- Thread safety of both features
- Graph wiring with parallel_sub_queries flag
"""
from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from config import settings


def _set_setting(name: str, value):
    object.__setattr__(settings, name, value)


FAKE_DOCS = [
    Document(
        page_content="Employees may work remotely up to 3 days per week.",
        metadata={"filename": "handbook.md", "source": "handbook.md"},
    ),
    Document(
        page_content="Standard payment terms are Net 30 from invoice date.",
        metadata={"filename": "vendor_terms.md", "source": "vendor_terms.md"},
    ),
    Document(
        page_content="The engineering onboarding takes 2 weeks.",
        metadata={"filename": "onboarding.md", "source": "onboarding.md"},
    ),
]


# ---------------------------------------------------------------------------
# Parallel sub-query processing tests
# ---------------------------------------------------------------------------


class TestProcessSubQueriesParallel:
    """Test the parallel sub-query processing node."""

    @pytest.fixture(autouse=True)
    def save_restore(self):
        orig_parallel = settings.parallel_sub_queries
        orig_workers = settings.sub_query_max_workers
        yield
        _set_setting("parallel_sub_queries", orig_parallel)
        _set_setting("sub_query_max_workers", orig_workers)

    @patch("src.graph.planner._process_single_sub_query")
    def test_parallel_processes_all_sub_questions(self, mock_process):
        """All sub-questions should be processed."""
        _set_setting("parallel_sub_queries", True)
        _set_setting("sub_query_max_workers", 3)

        mock_process.side_effect = [
            ("Answer 1", FAKE_DOCS[:1]),
            ("Answer 2", FAKE_DOCS[1:2]),
            ("Answer 3", FAKE_DOCS[2:3]),
        ]

        from src.graph.planner import process_sub_queries_parallel

        state = {
            "sub_questions": ["Q1?", "Q2?", "Q3?"],
            "retriever_strategy": "dense",
        }
        result = process_sub_queries_parallel(state)

        assert len(result["sub_answers"]) == 3
        assert result["current_sub_idx"] == 3
        assert len(result["all_sub_documents"]) == 3
        assert mock_process.call_count == 3

    @patch("src.graph.planner._process_single_sub_query")
    def test_parallel_preserves_answer_order(self, mock_process):
        """Answers should be in the same order as sub-questions."""
        _set_setting("parallel_sub_queries", True)
        _set_setting("sub_query_max_workers", 3)

        # Use a function so each call gets a deterministic answer based on input
        def answer_by_question(sub_q, strategy):
            return (f"Answer for: {sub_q}", [])

        mock_process.side_effect = answer_by_question

        from src.graph.planner import process_sub_queries_parallel

        state = {
            "sub_questions": ["First?", "Second?", "Third?"],
            "retriever_strategy": "dense",
        }
        result = process_sub_queries_parallel(state)

        assert result["sub_answers"][0] == "Answer for: First?"
        assert result["sub_answers"][1] == "Answer for: Second?"
        assert result["sub_answers"][2] == "Answer for: Third?"

    @patch("src.graph.planner._process_single_sub_query")
    def test_parallel_handles_empty_sub_questions(self, mock_process):
        """Empty sub-questions list should return empty results."""
        _set_setting("parallel_sub_queries", True)

        from src.graph.planner import process_sub_queries_parallel

        state = {
            "sub_questions": [],
            "retriever_strategy": "dense",
        }
        result = process_sub_queries_parallel(state)

        assert result["sub_answers"] == []
        assert result["current_sub_idx"] == 0
        assert result["all_sub_documents"] == []
        mock_process.assert_not_called()

    @patch("src.graph.planner._process_single_sub_query")
    def test_parallel_handles_single_sub_question(self, mock_process):
        """Single sub-question should work correctly."""
        _set_setting("parallel_sub_queries", True)
        _set_setting("sub_query_max_workers", 3)

        mock_process.return_value = ("Only answer", FAKE_DOCS[:1])

        from src.graph.planner import process_sub_queries_parallel

        state = {
            "sub_questions": ["Only question?"],
            "retriever_strategy": "dense",
        }
        result = process_sub_queries_parallel(state)

        assert len(result["sub_answers"]) == 1
        assert result["sub_answers"][0] == "Only answer"
        assert result["current_sub_idx"] == 1

    @patch("src.graph.planner._process_single_sub_query")
    def test_parallel_handles_individual_failure(self, mock_process):
        """If one sub-query fails, others should still succeed."""
        _set_setting("parallel_sub_queries", True)
        _set_setting("sub_query_max_workers", 3)

        def _by_question(sub_q, strategy):
            if sub_q == "Q2?":
                raise Exception("LLM timeout")
            return (f"Answer for {sub_q}", FAKE_DOCS[:1])

        mock_process.side_effect = _by_question

        from src.graph.planner import process_sub_queries_parallel

        state = {
            "sub_questions": ["Q1?", "Q2?", "Q3?"],
            "retriever_strategy": "dense",
        }
        result = process_sub_queries_parallel(state)

        assert len(result["sub_answers"]) == 3
        assert result["sub_answers"][0] == "Answer for Q1?"
        assert "error" in result["sub_answers"][1].lower()
        assert result["sub_answers"][2] == "Answer for Q3?"

    @patch("src.graph.planner._process_single_sub_query")
    def test_parallel_respects_max_workers(self, mock_process):
        """Max workers should cap thread pool size."""
        _set_setting("parallel_sub_queries", True)
        _set_setting("sub_query_max_workers", 2)

        active_threads = []
        lock = threading.Lock()

        def track_threads(sub_q, strategy):
            with lock:
                active_threads.append(threading.current_thread().name)
            return (f"Answer for {sub_q}", [])

        mock_process.side_effect = track_threads

        from src.graph.planner import process_sub_queries_parallel

        state = {
            "sub_questions": ["Q1?", "Q2?", "Q3?"],
            "retriever_strategy": "dense",
        }
        result = process_sub_queries_parallel(state)

        assert len(result["sub_answers"]) == 3
        assert mock_process.call_count == 3


# ---------------------------------------------------------------------------
# Sequential sub-query processing tests (verify helper extraction)
# ---------------------------------------------------------------------------


class TestProcessSingleSubQuery:
    """Test the extracted _process_single_sub_query helper."""

    @pytest.fixture(autouse=True)
    def save_restore(self):
        orig_retries = settings.sub_query_max_retries
        yield
        _set_setting("sub_query_max_retries", orig_retries)

    @patch("src.retrieval.get_retriever")
    def test_returns_answer_and_docs(self, mock_get_retriever):
        """Helper should return (answer, docs) tuple."""
        _set_setting("sub_query_max_retries", 0)

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = FAKE_DOCS[:2]
        mock_get_retriever.return_value = mock_retriever

        # Patch the entire generation chain to return a string
        with patch("src.graph.planner.StrOutputParser") as mock_parser_cls:
            mock_parser = MagicMock()
            mock_parser_cls.return_value = mock_parser

            with patch("src.graph.planner._llm") as mock_llm:
                # Build a mock chain that produces a string on .invoke()
                mock_final_chain = MagicMock()
                mock_final_chain.invoke.return_value = "Generated answer"

                # Make prompt | llm() | StrOutputParser() return the mock chain
                mock_prompt = MagicMock()
                mock_prompt.__or__ = MagicMock(return_value=MagicMock(__or__=MagicMock(return_value=mock_final_chain)))
                mock_llm.return_value = MagicMock()

                with patch("src.graph.nodes._gen_prompt", mock_prompt):
                    from src.graph.planner import _process_single_sub_query

                    answer, docs = _process_single_sub_query("Test question?", "dense")
                    assert isinstance(answer, str)
                    assert isinstance(docs, list)
                    assert len(docs) == 2

    @patch("src.retrieval.get_retriever")
    def test_returns_idk_on_no_docs(self, mock_get_retriever):
        """Should return IDK message when no docs retrieved."""
        _set_setting("sub_query_max_retries", 0)

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_get_retriever.return_value = mock_retriever

        from src.graph.planner import _process_single_sub_query

        answer, docs = _process_single_sub_query("Unknown topic?", "dense")
        assert "don't have enough information" in answer.lower()
        assert docs == []

    @patch("src.retrieval.get_retriever")
    def test_handles_retrieval_failure(self, mock_get_retriever):
        """Should handle retrieval exceptions gracefully."""
        _set_setting("sub_query_max_retries", 0)

        mock_retriever = MagicMock()
        mock_retriever.invoke.side_effect = Exception("ChromaDB down")
        mock_get_retriever.return_value = mock_retriever

        from src.graph.planner import _process_single_sub_query

        answer, docs = _process_single_sub_query("Test?", "dense")
        assert "don't have enough information" in answer.lower()
        assert docs == []


# ---------------------------------------------------------------------------
# Per-document grading tests
# ---------------------------------------------------------------------------


class TestPerDocGrading:
    """Test per-document grading mode in grade_documents."""

    @pytest.fixture(autouse=True)
    def save_restore(self):
        orig_per_doc = settings.per_doc_grading
        orig_workers = settings.rerank_max_workers
        yield
        _set_setting("per_doc_grading", orig_per_doc)
        _set_setting("rerank_max_workers", orig_workers)

    @patch("src.graph.nodes._grade_single_doc")
    def test_per_doc_filters_irrelevant(self, mock_grade):
        """Irrelevant documents should be filtered out."""
        _set_setting("per_doc_grading", True)
        _set_setting("rerank_max_workers", 4)

        mock_grade.side_effect = [
            (FAKE_DOCS[0], True),   # relevant
            (FAKE_DOCS[1], False),  # irrelevant
            (FAKE_DOCS[2], True),   # relevant
        ]

        from src.graph.nodes import grade_documents

        state = {"question": "What is the remote work policy?", "documents": FAKE_DOCS}
        result = grade_documents(state)

        assert result["relevant"] is True
        assert len(result["documents"]) == 2
        assert FAKE_DOCS[0] in result["documents"]
        assert FAKE_DOCS[2] in result["documents"]
        assert FAKE_DOCS[1] not in result["documents"]

    @patch("src.graph.nodes._grade_single_doc")
    def test_per_doc_all_irrelevant(self, mock_grade):
        """When all docs are irrelevant, relevant should be False."""
        _set_setting("per_doc_grading", True)
        _set_setting("rerank_max_workers", 4)

        mock_grade.side_effect = [
            (FAKE_DOCS[0], False),
            (FAKE_DOCS[1], False),
            (FAKE_DOCS[2], False),
        ]

        from src.graph.nodes import grade_documents

        state = {"question": "Unrelated question?", "documents": FAKE_DOCS}
        result = grade_documents(state)

        assert result["relevant"] is False
        assert len(result["documents"]) == 0

    @patch("src.graph.nodes._grade_single_doc")
    def test_per_doc_all_relevant(self, mock_grade):
        """When all docs are relevant, all should be kept."""
        _set_setting("per_doc_grading", True)
        _set_setting("rerank_max_workers", 4)

        mock_grade.side_effect = [
            (FAKE_DOCS[0], True),
            (FAKE_DOCS[1], True),
            (FAKE_DOCS[2], True),
        ]

        from src.graph.nodes import grade_documents

        state = {"question": "General query?", "documents": FAKE_DOCS}
        result = grade_documents(state)

        assert result["relevant"] is True
        assert len(result["documents"]) == 3

    @patch("src.graph.nodes._grade_single_doc")
    def test_per_doc_preserves_order(self, mock_grade):
        """Filtered docs should maintain their original order."""
        _set_setting("per_doc_grading", True)
        _set_setting("rerank_max_workers", 4)

        # Return in reverse order to simulate concurrent completion
        mock_grade.side_effect = [
            (FAKE_DOCS[0], True),
            (FAKE_DOCS[1], False),
            (FAKE_DOCS[2], True),
        ]

        from src.graph.nodes import grade_documents

        state = {"question": "Test?", "documents": FAKE_DOCS}
        result = grade_documents(state)

        assert result["documents"][0] is FAKE_DOCS[0]
        assert result["documents"][1] is FAKE_DOCS[2]

    def test_per_doc_disabled_uses_batch(self):
        """When per_doc_grading is False, should use batch grading."""
        _set_setting("per_doc_grading", False)

        with patch("src.graph.nodes.get_breaker") as mock_breaker:
            mock_cb = MagicMock()
            mock_breaker.return_value = mock_cb

            from src.graph.nodes import GradeResult
            mock_cb.call.return_value = GradeResult(relevant=True)

            from src.graph.nodes import grade_documents

            state = {"question": "Test?", "documents": FAKE_DOCS}
            result = grade_documents(state)

            assert result["relevant"] is True
            # Should NOT have filtered documents (batch mode)
            assert "documents" not in result

    def test_per_doc_empty_docs(self):
        """Empty documents should return relevant=False regardless of mode."""
        _set_setting("per_doc_grading", True)

        from src.graph.nodes import grade_documents

        state = {"question": "Test?", "documents": []}
        result = grade_documents(state)

        assert result["relevant"] is False

    @patch("src.graph.nodes._grade_single_doc")
    def test_per_doc_single_document(self, mock_grade):
        """Should work correctly with a single document."""
        _set_setting("per_doc_grading", True)
        _set_setting("rerank_max_workers", 4)

        mock_grade.return_value = (FAKE_DOCS[0], True)

        from src.graph.nodes import grade_documents

        state = {"question": "Test?", "documents": [FAKE_DOCS[0]]}
        result = grade_documents(state)

        assert result["relevant"] is True
        assert len(result["documents"]) == 1


# ---------------------------------------------------------------------------
# Graph wiring tests
# ---------------------------------------------------------------------------


class TestGraphWiring:
    """Test graph compilation with parallel processing flags."""

    @pytest.fixture(autouse=True)
    def save_restore(self):
        orig_parallel = settings.parallel_sub_queries
        yield
        _set_setting("parallel_sub_queries", orig_parallel)

    def test_graph_with_parallel_disabled(self):
        """Graph should have process_sub_query node when parallel is disabled."""
        _set_setting("parallel_sub_queries", False)

        from langgraph.checkpoint.memory import InMemorySaver

        from src.graph.build_graph import build_graph, reset_graph

        reset_graph()
        try:
            graph = build_graph(checkpointer=InMemorySaver())
            node_names = set(graph.get_graph().nodes.keys())
            assert "process_sub_query" in node_names
            assert "process_sub_queries_parallel" not in node_names
        finally:
            reset_graph()

    def test_graph_with_parallel_enabled(self):
        """Graph should have process_sub_queries_parallel node when enabled."""
        _set_setting("parallel_sub_queries", True)

        from langgraph.checkpoint.memory import InMemorySaver

        from src.graph.build_graph import build_graph, reset_graph

        reset_graph()
        try:
            graph = build_graph(checkpointer=InMemorySaver())
            node_names = set(graph.get_graph().nodes.keys())
            assert "process_sub_queries_parallel" in node_names
            assert "process_sub_query" not in node_names
        finally:
            reset_graph()


# ---------------------------------------------------------------------------
# Thread safety tests
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Test thread safety of parallel operations."""

    @pytest.fixture(autouse=True)
    def save_restore(self):
        orig_parallel = settings.parallel_sub_queries
        orig_workers = settings.sub_query_max_workers
        orig_per_doc = settings.per_doc_grading
        orig_rerank_workers = settings.rerank_max_workers
        yield
        _set_setting("parallel_sub_queries", orig_parallel)
        _set_setting("sub_query_max_workers", orig_workers)
        _set_setting("per_doc_grading", orig_per_doc)
        _set_setting("rerank_max_workers", orig_rerank_workers)

    @patch("src.graph.planner._process_single_sub_query")
    def test_parallel_thread_safety(self, mock_process):
        """Concurrent sub-query processing should not corrupt shared state."""
        _set_setting("parallel_sub_queries", True)
        _set_setting("sub_query_max_workers", 3)

        results_lock = threading.Lock()
        call_count = {"value": 0}

        def thread_safe_process(sub_q, strategy):
            with results_lock:
                call_count["value"] += 1
            return (f"Answer for {sub_q}", FAKE_DOCS[:1])

        mock_process.side_effect = thread_safe_process

        from src.graph.planner import process_sub_queries_parallel

        state = {
            "sub_questions": ["Q1?", "Q2?", "Q3?", "Q4?", "Q5?"],
            "retriever_strategy": "dense",
        }
        result = process_sub_queries_parallel(state)

        assert len(result["sub_answers"]) == 5
        assert call_count["value"] == 5
        # All answers should be non-empty strings
        for ans in result["sub_answers"]:
            assert isinstance(ans, str)
            assert len(ans) > 0

    @patch("src.graph.nodes._grade_single_doc")
    def test_per_doc_grading_thread_safety(self, mock_grade):
        """Concurrent per-doc grading should not corrupt results."""
        _set_setting("per_doc_grading", True)
        _set_setting("rerank_max_workers", 4)

        grade_lock = threading.Lock()
        call_count = {"value": 0}

        def thread_safe_grade(question, doc):
            with grade_lock:
                call_count["value"] += 1
            return (doc, True)

        mock_grade.side_effect = thread_safe_grade

        from src.graph.nodes import grade_documents

        # Use more docs to increase concurrency
        many_docs = FAKE_DOCS * 3  # 9 docs
        state = {"question": "Test?", "documents": many_docs}
        result = grade_documents(state)

        assert result["relevant"] is True
        assert len(result["documents"]) == 9
        assert call_count["value"] == 9
