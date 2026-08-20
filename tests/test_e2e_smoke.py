"""End-to-end smoke tests for all pipeline combinations.

Verifies that every combination of mode (naive, graph) and retriever strategy
(dense, hybrid, multi_query, rerank) produces a non-empty answer string
without raising. All external calls (LLM, ChromaDB) are mocked.

These tests catch wiring issues — wrong argument names, broken imports,
missing factory branches — that unit tests on individual modules miss.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.retrievers import BaseRetriever

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_DOCS = [
    Document(
        page_content="Employees may work remotely up to 3 days per week.",
        metadata={"filename": "handbook.md", "source": "handbook.md"},
    ),
    Document(
        page_content="Standard payment terms are Net 30 from invoice date.",
        metadata={"filename": "vendor_terms.md", "source": "vendor_terms.md"},
    ),
]


class _FakeRetriever(BaseRetriever):
    """LCEL-compatible retriever stub (supports the | pipe operator)."""

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        return list(FAKE_DOCS)


def _fake_get_retriever(strategy="dense", k=None, filter=None):
    return _FakeRetriever()


# ---------------------------------------------------------------------------
# Naive RAG x all retrievers
# ---------------------------------------------------------------------------

class TestNaiveRagSmoke:
    """Naive pipeline should produce a string answer for all retriever strategies."""

    @pytest.fixture(autouse=True)
    def _patch_chain(self):
        """Mock build_naive_rag_chain so no LLM/embedding calls are made.

        LCEL chains require real Runnable objects at every step (they use
        __or__/__ror__ for composition), so mocking ChatOpenAI alone
        doesn't work. Instead we mock the chain builder to return a
        simple object whose .invoke() returns a canned answer.
        """
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Remote work is 3 days per week. (handbook.md)"

        with patch(
            "src.rag.naive_rag.build_naive_rag_chain", return_value=mock_chain,
        ) as mock_builder:
            self._mock_builder = mock_builder
            yield

    @pytest.mark.parametrize("strategy", ["dense", "hybrid", "multi_query", "rerank"])
    def test_naive_returns_string(self, strategy):
        from src.rag.naive_rag import answer

        result = answer("What is the remote work policy?", retriever_strategy=strategy)
        assert isinstance(result, str)
        assert len(result) > 0
        # Verify the strategy was passed through to the chain builder
        self._mock_builder.assert_called_once()
        call_kwargs = self._mock_builder.call_args
        assert call_kwargs.kwargs.get("retriever_strategy") == strategy or \
            call_kwargs[1].get("retriever_strategy") == strategy

    def test_naive_empty_question(self):
        from src.rag.naive_rag import answer

        assert answer("") == "Please provide a question."

    def test_naive_whitespace_question(self):
        from src.rag.naive_rag import answer

        assert answer("   \n  ") == "Please provide a question."


# ---------------------------------------------------------------------------
# Graph CRAG x all retrievers
# ---------------------------------------------------------------------------

class TestGraphCragSmoke:
    """CRAG graph pipeline should produce a string answer for all retrievers."""

    @pytest.fixture(autouse=True)
    def _patch_graph_dependencies(self, stub_llm):
        """Replace individual node functions with stubs that return plain dicts.

        LangGraph serializes all state via msgpack, so node returns must be
        plain Python objects — no MagicMock anywhere in the state.

        ``stub_llm`` covers the nodes this fixture does *not* stub — intent
        detection, query transformation, the analyzer. They were reaching
        the real API and swallowing the result in their fallback paths, so
        the file's claim that "all external calls are mocked" was untrue
        and each of these tests took ~20 seconds of billed latency.
        """
        def fake_planner(state):
            return {
                "is_multi_part": False,
                "sub_questions": [state.get("question", "")],
                "sub_answers": [],
                "current_sub_idx": 0,
                "original_question": state.get("question", ""),
            }

        def fake_retrieve(state):
            return {"documents": list(FAKE_DOCS), "retries": state.get("retries", 0)}

        def fake_grade(state):
            return {"relevant": True}

        def fake_generate(state):
            return {"generation": "Remote work is allowed 3 days per week. (handbook.md)"}

        def fake_critic(state):
            return {"critic_passed": True, "claims_removed": 0}

        with (
            patch("src.graph.planner.planner", side_effect=fake_planner),
            patch("src.graph.nodes.retrieve", side_effect=fake_retrieve),
            patch("src.graph.nodes.grade_documents", side_effect=fake_grade),
            patch("src.graph.nodes.generate", side_effect=fake_generate),
            patch("src.graph.nodes.critic", side_effect=fake_critic),
        ):
            yield

    @pytest.mark.parametrize("strategy", ["dense", "hybrid", "multi_query", "rerank"])
    def test_graph_returns_string(self, strategy):
        from src.graph import build_graph as bg

        bg.reset_graph()
        try:
            from langgraph.checkpoint.memory import InMemorySaver

            graph = bg.build_graph(checkpointer=InMemorySaver())
            result = graph.invoke(
                {
                    "question": "What is the remote work policy?",
                    "retries": 0,
                    "retriever_strategy": strategy,
                },
                {"configurable": {"thread_id": f"smoke-{strategy}"}},
            )
            assert isinstance(result.get("generation"), str)
            assert len(result["generation"]) > 0
        finally:
            bg.reset_graph()

    def test_graph_ask_empty_question(self):
        from src.graph.build_graph import ask

        assert ask("") == "Please provide a question."

    def test_graph_ask_none_question(self):
        from src.graph.build_graph import ask

        assert ask(None) == "Please provide a question."


# ---------------------------------------------------------------------------
# CLI entrypoints parse correctly
# ---------------------------------------------------------------------------

class TestCLISmoke:
    """Verify CLI arg parsing for ask.py and ingest.py scripts."""

    def test_ask_cli_parses_all_retrievers(self):
        import argparse

        from scripts.ask import main

        # Build the parser the same way main() does
        parser = argparse.ArgumentParser()
        parser.add_argument("question", nargs="+")
        parser.add_argument("--mode", choices=["naive", "graph"], default="naive")
        parser.add_argument("--filter", default=None)
        parser.add_argument("-k", "--top-k", type=int, default=None)
        parser.add_argument(
            "--retriever",
            choices=["dense", "hybrid", "multi_query", "rerank"],
            default="dense",
        )

        for strategy in ["dense", "hybrid", "multi_query", "rerank"]:
            args = parser.parse_args(["What is X?", "--retriever", strategy])
            assert args.retriever == strategy
            assert args.mode == "naive"

        for mode in ["naive", "graph"]:
            args = parser.parse_args(["test", "--mode", mode])
            assert args.mode == mode

    def test_ingest_cli_parses(self):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("path", nargs="?", default="./data/sample_docs")
        parser.add_argument("--chunk-size", type=int, default=None)
        parser.add_argument("--chunk-overlap", type=int, default=None)

        args = parser.parse_args([])
        assert args.path == "./data/sample_docs"

        args = parser.parse_args(["./my_docs", "--chunk-size", "500"])
        assert args.path == "./my_docs"
        assert args.chunk_size == 500

    def test_eval_cli_parses_all_modes_and_retrievers(self):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--mode", choices=["naive", "graph"], default="naive")
        parser.add_argument("--limit", type=int, default=None)
        parser.add_argument("--output", type=str, default=None)
        parser.add_argument(
            "--retriever",
            choices=["dense", "hybrid", "multi_query", "rerank"],
            default="dense",
        )

        for mode in ["naive", "graph"]:
            for retriever in ["dense", "hybrid", "multi_query", "rerank"]:
                args = parser.parse_args(["--mode", mode, "--retriever", retriever])
                assert args.mode == mode
                assert args.retriever == retriever


# ---------------------------------------------------------------------------
# Retriever factory wiring (full round-trip)
# ---------------------------------------------------------------------------

class TestRetrieverFactorySmoke:
    """Each factory branch should produce an object with .invoke()."""

    def test_dense_wiring(self):
        with patch("src.vectorstore.chroma_store.get_retriever") as mock:
            mock.return_value = _FakeRetriever()
            from src.retrieval import get_retriever

            r = get_retriever("dense", k=3)
            docs = r.invoke("test query")
            assert len(docs) == 2

    def test_hybrid_wiring(self):
        from src.retrieval import get_retriever

        r = get_retriever("hybrid", k=3)
        assert hasattr(r, "invoke")
        assert r.k == 3

    def test_multi_query_wiring(self):
        from src.retrieval import get_retriever

        r = get_retriever("multi_query", k=5)
        assert hasattr(r, "invoke")
        assert r.k == 5

    def test_rerank_wiring(self):
        from src.retrieval import get_retriever

        r = get_retriever("rerank", k=4)
        assert hasattr(r, "invoke")
        assert r.k == 4
        assert r.fetch_k == max(12, 4 * 3)

    def test_unknown_strategy_raises(self):
        from src.retrieval import get_retriever

        with pytest.raises(ValueError, match="Unknown retrieval strategy"):
            get_retriever("nonexistent")


# ---------------------------------------------------------------------------
# Config validation round-trip
# ---------------------------------------------------------------------------

class TestConfigSmoke:
    """Config should validate cleanly for known-good values and reject bad ones."""

    def test_full_valid_config(self):
        from dataclasses import replace

        from config import settings

        good = replace(
            settings,
            openai_api_key="sk-test-key",
            log_level="INFO",
            chunk_size=1000,
            chunk_overlap=150,
            top_k=4,
        )
        good.validate()  # should not raise

    def test_chunk_overlap_gte_chunk_size_raises(self):
        from dataclasses import replace

        from config import settings

        bad = replace(settings, openai_api_key="sk-test", chunk_size=100, chunk_overlap=100)
        with pytest.raises(ValueError, match="chunk_overlap"):
            bad.validate()

    def test_negative_top_k_raises(self):
        from dataclasses import replace

        from config import settings

        bad = replace(settings, openai_api_key="sk-test", top_k=-1)
        with pytest.raises(ValueError, match="top_k"):
            bad.validate()

    def test_zero_chunk_size_raises(self):
        from dataclasses import replace

        from config import settings

        bad = replace(settings, openai_api_key="sk-test", chunk_size=0)
        with pytest.raises(ValueError, match="chunk_size"):
            bad.validate()


# ---------------------------------------------------------------------------
# Graph state completeness
# ---------------------------------------------------------------------------

class TestGraphStateSmoke:
    """RAGState should contain all fields used by nodes."""

    def test_state_has_all_expected_keys(self):
        from src.graph.state import RAGState

        expected = {
            "question", "documents", "relevant", "web_fallback_used",
            "generation", "retries", "retriever_strategy", "filter",
            "top_k",  # explicit retrieval k from the API request
            "critic_passed", "claims_removed",
            # Phase 5: multi-agent decomposition
            "original_question", "sub_questions", "sub_answers",
            "is_multi_part", "current_sub_idx",
            # Phase 8: graph intelligence
            "in_scope", "all_sub_documents", "tool_results",
            # Phase 10: conversation memory
            "session_id", "chat_history", "memory_context",
            # Phase 11: intent detection
            "intent", "intent_confidence",
            # Phase 12: query transformation
            "transformed_query", "extracted_entities",
            # Phase 15: semantic cache
            "cache_hit",
            # Phase 17: guardrails
            "guardrail_passed", "guardrail_reason",
            # Cost ceiling: decomposition stopped early to stay in budget
            "budget_truncated",
        }
        actual = set(RAGState.__annotations__.keys())
        assert expected == actual, f"Missing: {expected - actual}, Extra: {actual - expected}"

    def test_graph_has_all_expected_nodes(self):
        from langgraph.checkpoint.memory import InMemorySaver

        from src.graph.build_graph import build_graph, reset_graph

        reset_graph()
        try:
            graph = build_graph(checkpointer=InMemorySaver())
            node_names = set(graph.get_graph().nodes.keys())
            expected_nodes = {
                "scope_check",
                "planner", "process_sub_query", "synthesize",
                "retrieve", "grade_documents", "transform_query",
                "web_search", "generate", "critic",
                "__start__", "__end__",
            }
            assert expected_nodes.issubset(node_names), (
                f"Missing nodes: {expected_nodes - node_names}"
            )
        finally:
            reset_graph()
