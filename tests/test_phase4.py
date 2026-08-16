"""Phase 4 tests: critic node, graph integration, tracing, SQLite checkpointer.

Tests cover:
  - Critic node: all claims supported, some unsupported, all unsupported, IDK passthrough
  - Graph structure: critic node wired in, expected node set
  - Tracing: decorator preserves function behavior
  - SQLite checkpointer: graph compiles with SQLite
  - Integration: full graph flow with mocked LLM (relevant path, retry path)
"""
from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.graph.nodes import (
    ClaimVerdict,
    critic,
    decide_after_grade,
    generate,
    retrieve,
    web_search,
)
from src.graph.tracing import traced

# === Critic node tests ===

class TestCritic:
    def _make_state(self, generation: str, docs: list[Document] | None = None) -> dict:
        return {
            "question": "What is the password policy?",
            "generation": generation,
            "documents": docs or [
                Document(
                    page_content="Passwords must be at least 14 characters.",
                    metadata={"filename": "security_policy.md"},
                ),
            ],
        }

    @patch("src.graph.nodes._llm")
    def test_all_claims_supported(self, mock_llm_factory):
        """When all claims are supported, critic passes through unchanged."""
        mock_llm = MagicMock()
        mock_llm_factory.return_value = mock_llm
        mock_llm.with_structured_output.return_value.invoke.return_value = ClaimVerdict(
            supported_claims=["Passwords must be at least 14 characters"],
            unsupported_claims=[],
        )

        state = self._make_state("Passwords must be at least 14 characters (security_policy.md).")
        result = critic(state)

        assert result["critic_passed"] is True
        assert result["claims_removed"] == 0
        # generation should NOT be in result (unchanged)
        assert "generation" not in result

    def test_some_claims_unsupported_triggers_rewrite(self):
        """When some claims are unsupported, critic rewrites the answer."""
        mock_verifier = MagicMock()
        mock_verifier.invoke.return_value = ClaimVerdict(
            supported_claims=["Passwords must be 14 chars"],
            unsupported_claims=["Passwords must be changed every 30 days"],
        )

        mock_rewriter = MagicMock()
        mock_rewriter.invoke.return_value = "Passwords must be 14 characters (security_policy.md)."

        with patch("src.graph.nodes._critic_prompt") as mock_cp, \
             patch("src.graph.nodes._rewrite_prompt_critic") as mock_rp, \
             patch("src.graph.nodes._llm") as mock_llm_factory:
            # Verifier chain: _critic_prompt | _llm().with_structured_output(...)
            mock_llm = MagicMock()
            mock_llm_factory.return_value = mock_llm
            mock_cp.__or__ = MagicMock(return_value=mock_verifier)
            # Rewriter chain: _rewrite_prompt_critic | _llm() | StrOutputParser()
            mock_rp.__or__ = MagicMock(return_value=mock_rewriter)

            state = self._make_state(
                "Passwords must be 14 chars. They must be changed every 30 days."
            )
            result = critic(state)

        assert result["critic_passed"] is False
        assert result["claims_removed"] == 1

    def test_no_supported_claims_returns_idk(self):
        """When no claims are supported, critic returns IDK."""
        mock_verifier = MagicMock()
        mock_verifier.invoke.return_value = ClaimVerdict(
            supported_claims=[],
            unsupported_claims=["The sky is blue", "Water is wet"],
        )

        with patch("src.graph.nodes._critic_prompt") as mock_cp, \
             patch("src.graph.nodes._llm") as mock_llm_factory:
            mock_llm = MagicMock()
            mock_llm_factory.return_value = mock_llm
            mock_cp.__or__ = MagicMock(return_value=mock_verifier)

            state = self._make_state("The sky is blue. Water is wet.")
            result = critic(state)

        assert result["critic_passed"] is False
        assert result["claims_removed"] == 2
        assert "don't have enough information" in result["generation"].lower()

    def test_idk_answer_skips_critic(self):
        """IDK answers should pass through without LLM calls."""
        state = self._make_state(
            "I don't have enough information in the available documents to answer this question."
        )
        result = critic(state)
        assert result["critic_passed"] is True
        assert result["claims_removed"] == 0

    def test_empty_documents_skips_critic(self):
        """No source documents means nothing to verify against."""
        state = {
            "question": "test",
            "generation": "Some answer",
            "documents": [],
        }
        result = critic(state)
        assert result["critic_passed"] is True
        assert result["claims_removed"] == 0

    @patch("src.graph.nodes._llm")
    def test_critic_error_passes_through(self, mock_llm_factory):
        """If the critic LLM call fails, answer passes through unchanged."""
        mock_llm = MagicMock()
        mock_llm_factory.return_value = mock_llm
        mock_llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("LLM down")

        state = self._make_state("Some answer text")
        result = critic(state)

        assert result["critic_passed"] is True
        assert result["claims_removed"] == 0


# === ClaimVerdict model tests ===

class TestClaimVerdict:
    def test_all_supported(self):
        v = ClaimVerdict(supported_claims=["a", "b"], unsupported_claims=[])
        assert len(v.supported_claims) == 2
        assert len(v.unsupported_claims) == 0

    def test_mixed(self):
        v = ClaimVerdict(supported_claims=["a"], unsupported_claims=["b"])
        assert len(v.supported_claims) == 1
        assert len(v.unsupported_claims) == 1

    def test_empty(self):
        v = ClaimVerdict(supported_claims=[], unsupported_claims=[])
        assert len(v.supported_claims) == 0


# === Tracing tests ===

class TestTracing:
    def test_traced_preserves_function_name(self):
        @traced
        def my_node(state: dict) -> dict:
            return {"result": True}

        assert my_node.__name__ == "my_node"

    def test_traced_passes_through_result(self):
        @traced
        def my_node(state: dict) -> dict:
            return {"answer": "hello", "documents": []}

        result = my_node({"question": "test"})
        assert result == {"answer": "hello", "documents": []}

    def test_traced_handles_empty_state(self):
        @traced
        def my_node(state: dict) -> dict:
            return {}

        result = my_node({})
        assert result == {}


# === Graph structure tests ===

class TestGraphWithCritic:
    def test_graph_has_critic_node(self):
        from langgraph.checkpoint.memory import InMemorySaver

        from src.graph.build_graph import build_graph

        graph = build_graph(checkpointer=InMemorySaver())
        node_names = set(graph.get_graph().nodes.keys())
        expected = {"retrieve", "grade_documents", "transform_query", "web_search", "generate", "critic"}
        assert expected.issubset(node_names)

    def test_graph_compiles_with_sqlite(self):
        from langgraph.checkpoint.sqlite import SqliteSaver

        from src.graph.build_graph import build_graph

        conn = sqlite3.connect(":memory:")
        saver = SqliteSaver(conn)
        graph = build_graph(checkpointer=saver)
        assert graph is not None
        conn.close()


# === Integration tests (mocked LLM, full graph flow) ===

class TestGraphIntegration:
    @patch("src.graph.nodes.get_retriever")
    @patch("src.graph.nodes._critic_prompt")
    @patch("src.graph.nodes._gen_prompt")
    @patch("src.graph.nodes._grade_prompt")
    @patch("src.graph.nodes._llm")
    def test_relevant_path_with_critic(
        self, mock_llm_factory, mock_grade_prompt, mock_gen_prompt,
        mock_critic_prompt, mock_get_retriever,
    ):
        """Test: retrieve → grade (relevant) → generate → critic → END."""
        from langgraph.checkpoint.memory import InMemorySaver

        from src.graph.build_graph import build_graph
        from src.graph.nodes import GradeResult

        # Mock retriever — returns real Documents
        docs = [Document(page_content="Policy says 14 chars min.", metadata={"filename": "sec.md"})]
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = docs
        mock_get_retriever.return_value = mock_retriever

        # Mock LLM
        mock_llm = MagicMock()
        mock_llm_factory.return_value = mock_llm

        # Mock grade chain: _grade_prompt | _llm().with_structured_output(GradeResult)
        mock_grade_chain = MagicMock()
        mock_grade_chain.invoke.return_value = GradeResult(relevant=True)
        mock_grade_prompt.__or__ = MagicMock(return_value=mock_grade_chain)

        # Mock generate chain: _gen_prompt | _llm() | StrOutputParser()
        mock_gen_chain = MagicMock()
        mock_gen_chain.__or__ = MagicMock(return_value=mock_gen_chain)
        mock_gen_chain.invoke.return_value = "Password minimum is 14 characters (sec.md)."
        mock_gen_prompt.__or__ = MagicMock(return_value=mock_gen_chain)

        # Mock critic chain: _critic_prompt | _llm().with_structured_output(ClaimVerdict)
        mock_critic_chain = MagicMock()
        mock_critic_chain.invoke.return_value = ClaimVerdict(
            supported_claims=["Password minimum is 14 characters"],
            unsupported_claims=[],
        )
        mock_critic_prompt.__or__ = MagicMock(return_value=mock_critic_chain)

        graph = build_graph(checkpointer=InMemorySaver())
        result = graph.invoke(
            {"question": "What is the password policy?", "retries": 0},
            {"configurable": {"thread_id": "test-relevant"}},
        )

        assert result.get("relevant") is True
        assert result.get("critic_passed") is True
        assert "14 characters" in result.get("generation", "")

    @patch("src.graph.nodes.get_retriever")
    @patch("src.graph.nodes._critic_prompt")
    @patch("src.graph.nodes._gen_prompt")
    @patch("src.graph.nodes._grade_prompt")
    @patch("src.graph.nodes._rewrite_prompt")
    @patch("src.graph.nodes._llm")
    def test_retry_path_with_critic(
        self, mock_llm_factory, mock_rewrite_prompt, mock_grade_prompt,
        mock_gen_prompt, mock_critic_prompt, mock_get_retriever,
    ):
        """Test: retrieve → grade (not relevant) → rewrite → retrieve → grade (relevant) → generate → critic."""
        from langgraph.checkpoint.memory import InMemorySaver

        from src.graph.build_graph import build_graph
        from src.graph.nodes import GradeResult

        # Mock retriever
        docs = [Document(page_content="Info about passwords.", metadata={"filename": "sec.md"})]
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = docs
        mock_get_retriever.return_value = mock_retriever

        mock_llm = MagicMock()
        mock_llm_factory.return_value = mock_llm

        # Grade: first call returns not relevant, second returns relevant
        mock_grade_chain = MagicMock()
        mock_grade_chain.invoke.side_effect = [
            GradeResult(relevant=False),
            GradeResult(relevant=True),
        ]
        mock_grade_prompt.__or__ = MagicMock(return_value=mock_grade_chain)

        # Rewrite chain
        mock_rewrite_chain = MagicMock()
        mock_rewrite_chain.__or__ = MagicMock(return_value=mock_rewrite_chain)
        mock_rewrite_chain.invoke.return_value = "What is the password policy?"
        mock_rewrite_prompt.__or__ = MagicMock(return_value=mock_rewrite_chain)

        # Generate
        mock_gen_chain = MagicMock()
        mock_gen_chain.__or__ = MagicMock(return_value=mock_gen_chain)
        mock_gen_chain.invoke.return_value = "Passwords info (sec.md)."
        mock_gen_prompt.__or__ = MagicMock(return_value=mock_gen_chain)

        # Critic
        mock_critic_chain = MagicMock()
        mock_critic_chain.invoke.return_value = ClaimVerdict(
            supported_claims=["Passwords info"],
            unsupported_claims=[],
        )
        mock_critic_prompt.__or__ = MagicMock(return_value=mock_critic_chain)

        graph = build_graph(checkpointer=InMemorySaver())
        result = graph.invoke(
            {"question": "password stuff", "retries": 0},
            {"configurable": {"thread_id": "test-retry"}},
        )

        assert result.get("retries") == 1  # had to retry once
        assert result.get("relevant") is True
        assert result.get("critic_passed") is True

    def test_empty_question_through_ask(self):
        from src.graph.build_graph import ask
        result = ask("")
        assert result == "Please provide a question."
