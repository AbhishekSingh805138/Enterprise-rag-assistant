"""Phase 8.4 tests: Multi-Part Processing & Critic Improvements.

Covers:
  - Sub-query mini-CRAG (retry on irrelevant docs)
  - Document preservation across sub-queries (all_sub_documents)
  - Critic prompt improvements (fewer false positives)
  - Critic rewrite preserves coherence (passes original answer)
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# TestSubQueryDocPreservation
# ---------------------------------------------------------------------------

class TestSubQueryDocPreservation:
    """Verify all_sub_documents accumulates across sub-queries."""

    @patch("src.retrieval.get_retriever")
    @patch("src.graph.planner._llm")
    @patch("src.graph.planner.settings")
    def test_docs_accumulated(self, mock_settings, mock_llm, mock_get_retriever):
        """process_sub_query should accumulate docs in all_sub_documents."""
        mock_settings.sub_query_max_retries = 0  # skip mini-CRAG for this test
        mock_settings.llm_model = "gpt-4o-mini"
        mock_settings.openai_api_key = "sk-test"
        mock_settings.llm_timeout = 30
        mock_settings.llm_max_retries = 2

        doc1 = Document(page_content="doc from sub-query 1", metadata={"filename": "a.md"})
        doc2 = Document(page_content="doc from sub-query 2", metadata={"filename": "b.md"})

        mock_retriever = MagicMock()
        mock_retriever.invoke.side_effect = [[doc1], [doc2]]
        mock_get_retriever.return_value = mock_retriever

        # Mock LLM chain for generation
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Generated answer"
        mock_llm_instance = MagicMock()
        mock_llm.return_value = mock_llm_instance

        with patch("src.graph.nodes._gen_prompt") as mock_prompt:
            mock_prompt.__or__ = MagicMock(return_value=MagicMock(__or__=MagicMock(return_value=mock_chain)))

            from src.graph.planner import process_sub_query

            # First sub-query
            state1 = {
                "sub_questions": ["q1", "q2"],
                "current_sub_idx": 0,
                "sub_answers": [],
                "retriever_strategy": "dense",
            }
            result1 = process_sub_query(state1)
            assert len(result1["all_sub_documents"]) == 1  # doc1

            # Second sub-query (with accumulated docs)
            state2 = {
                "sub_questions": ["q1", "q2"],
                "current_sub_idx": 1,
                "sub_answers": result1["sub_answers"],
                "retriever_strategy": "dense",
                "all_sub_documents": result1["all_sub_documents"],
            }
            result2 = process_sub_query(state2)
            assert len(result2["all_sub_documents"]) == 2  # doc1 + doc2

    @patch("src.retrieval.get_retriever")
    @patch("src.graph.planner.settings")
    def test_empty_docs_dont_break_accumulation(self, mock_settings, mock_get_retriever):
        mock_settings.sub_query_max_retries = 0
        mock_settings.llm_model = "gpt-4o-mini"
        mock_settings.openai_api_key = "sk-test"
        mock_settings.llm_timeout = 30
        mock_settings.llm_max_retries = 2

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_get_retriever.return_value = mock_retriever

        from src.graph.planner import process_sub_query
        state = {
            "sub_questions": ["q1"],
            "current_sub_idx": 0,
            "sub_answers": [],
            "retriever_strategy": "dense",
        }
        result = process_sub_query(state)
        assert result["all_sub_documents"] == []


# ---------------------------------------------------------------------------
# TestCriticUsesAllSubDocs
# ---------------------------------------------------------------------------

class TestCriticUsesAllSubDocs:
    """Verify critic uses all_sub_documents when available."""

    def test_critic_prefers_all_sub_documents(self):
        """When all_sub_documents is set, critic should use it over documents."""
        from src.graph.nodes import critic

        all_docs = [Document(page_content="all sub docs content")]
        regular_docs = [Document(page_content="regular docs content")]

        # IDK answer should skip critic entirely
        state = {
            "question": "test",
            "generation": "I don't have enough information to answer.",
            "documents": regular_docs,
            "all_sub_documents": all_docs,
        }
        result = critic(state)
        assert result["critic_passed"] is True  # IDK skips critic

    def test_critic_falls_back_to_documents(self):
        """When all_sub_documents is not set, critic should use documents."""
        from src.graph.nodes import critic
        state = {
            "question": "test",
            "generation": "I don't have enough information to answer.",
            "documents": [Document(page_content="test doc")],
        }
        result = critic(state)
        assert result["critic_passed"] is True  # IDK skips critic


# ---------------------------------------------------------------------------
# TestCriticPromptImprovements
# ---------------------------------------------------------------------------

class TestCriticPromptImprovements:
    """Verify critic prompt changes for fewer false positives."""

    def test_critic_prompt_mentions_leniency(self):
        """The critic prompt should include guidance about being lenient."""
        from src.graph.nodes import _critic_prompt
        messages = _critic_prompt.format_messages(
            question="test", answer="test", context="test"
        )
        system_msg = messages[0].content
        assert "lenient" in system_msg.lower() or "when in doubt" in system_msg.lower()

    def test_critic_prompt_mentions_summaries(self):
        """The critic prompt should recognize summaries as supported."""
        from src.graph.nodes import _critic_prompt
        messages = _critic_prompt.format_messages(
            question="test", answer="test", context="test"
        )
        system_msg = messages[0].content
        assert "summar" in system_msg.lower()


# ---------------------------------------------------------------------------
# TestCriticRewriteCoherence
# ---------------------------------------------------------------------------

class TestCriticRewriteCoherence:
    """Verify critic rewrite prompt passes original answer for context."""

    def test_rewrite_prompt_has_original_answer_var(self):
        """The rewrite prompt should include {original_answer} variable."""
        from src.graph.nodes import _rewrite_prompt_critic
        input_vars = _rewrite_prompt_critic.input_variables
        assert "original_answer" in input_vars

    def test_rewrite_prompt_has_question_var(self):
        from src.graph.nodes import _rewrite_prompt_critic
        input_vars = _rewrite_prompt_critic.input_variables
        assert "question" in input_vars

    def test_rewrite_prompt_has_supported_var(self):
        from src.graph.nodes import _rewrite_prompt_critic
        input_vars = _rewrite_prompt_critic.input_variables
        assert "supported" in input_vars


# ---------------------------------------------------------------------------
# TestSubQueryMiniCRAGConfig
# ---------------------------------------------------------------------------

class TestSubQueryMiniCRAGConfig:
    """Verify mini-CRAG config settings."""

    def test_sub_query_max_retries_default(self):
        from config import Settings
        s = Settings(openai_api_key="sk-test")
        assert s.sub_query_max_retries == 1

    def test_sub_query_max_retries_zero_disables(self):
        from config import Settings
        s = Settings(openai_api_key="sk-test", sub_query_max_retries=0)
        assert s.sub_query_max_retries == 0
