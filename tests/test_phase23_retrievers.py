"""Phase 23 — ComposedRetriever and CrossEncoderRetriever.

Two-stage retrieval is where recall and precision are traded off, so the
assertions focus on the contract between stages: the first stage must
over-fetch, the second must narrow to exactly k, and a scoring failure on
one document must not discard it silently.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.retrieval.composed import ComposedRetriever, build_composed_retriever
from src.retrieval.cross_encoder_rerank import (
    CrossEncoderRetriever,
    _get_cross_encoder,
    build_cross_encoder_retriever,
    reset_cross_encoder,
)


def docs(n=6):
    return [
        Document(page_content=f"content {i}", metadata={"filename": f"f{i}.md"})
        for i in range(n)
    ]


@pytest.fixture(autouse=True)
def _reset_model():
    reset_cross_encoder()
    yield
    reset_cross_encoder()


@pytest.fixture
def fake_sentence_transformers(monkeypatch):
    """Inject a fake `sentence_transformers`.

    The package is an optional heavyweight dependency (torch); what is under
    test is how the loader caches and configures the model, not the model.
    """
    import sys
    import types

    module = types.ModuleType("sentence_transformers")
    module.CrossEncoder = MagicMock(name="CrossEncoder")
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return module


@pytest.fixture
def retrieval_settings():
    with (
        patch("src.retrieval.composed.settings") as cs,
        patch("src.retrieval.cross_encoder_rerank.settings") as xs,
    ):
        for s in (cs, xs):
            s.top_k = 4
            s.llm_model = "gpt-4o-mini"
            s.openai_api_key = "sk-test"
            s.llm_timeout = 30
            s.llm_max_retries = 2
            s.rerank_max_workers = 4
            s.cross_encoder_batch_size = 16
            s.cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            s.cross_encoder_device = "cpu"
            s.rerank_fetch_k = 12
        yield cs


# ---------------------------------------------------------------------------
# Cross-encoder model loading
# ---------------------------------------------------------------------------

class TestCrossEncoderModelLoading:
    def test_model_is_loaded_once_and_cached(
        self, retrieval_settings, fake_sentence_transformers
    ):
        """Reloading a cross-encoder per query would dominate retrieval latency."""
        first, second = _get_cross_encoder(), _get_cross_encoder()
        assert first is second
        fake_sentence_transformers.CrossEncoder.assert_called_once()

    def test_model_is_loaded_with_configured_name_and_device(
        self, retrieval_settings, fake_sentence_transformers
    ):
        _get_cross_encoder()
        fake_sentence_transformers.CrossEncoder.assert_called_once_with(
            "cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu"
        )

    def test_reset_forces_a_reload(self, retrieval_settings, fake_sentence_transformers):
        _get_cross_encoder()
        reset_cross_encoder()
        _get_cross_encoder()
        assert fake_sentence_transformers.CrossEncoder.call_count == 2

    def test_missing_sentence_transformers_gives_an_install_hint(self, retrieval_settings):
        import builtins

        real_import = builtins.__import__

        def _no_st(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("nope")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", _no_st):
            with pytest.raises(ImportError, match="pip install sentence-transformers"):
                _get_cross_encoder()


# ---------------------------------------------------------------------------
# CrossEncoderRetriever
# ---------------------------------------------------------------------------

class TestCrossEncoderRetriever:
    def test_returns_top_k_ordered_by_score(self, retrieval_settings):
        model = MagicMock()
        model.predict.return_value = [0.1, 0.9, 0.5, 0.7]
        r = CrossEncoderRetriever(k=2)
        with (
            patch.object(r, "_get_candidates", return_value=docs(4)),
            patch("src.retrieval.cross_encoder_rerank._get_cross_encoder", return_value=model),
        ):
            result = r.invoke("q")
        assert [d.page_content for d in result] == ["content 1", "content 3"]

    def test_scores_all_candidates_in_one_batch(self, retrieval_settings):
        """Batch scoring is the whole point — per-doc calls would be slower."""
        model = MagicMock()
        model.predict.return_value = [0.5] * 6
        r = CrossEncoderRetriever(k=3)
        with (
            patch.object(r, "_get_candidates", return_value=docs(6)),
            patch("src.retrieval.cross_encoder_rerank._get_cross_encoder", return_value=model),
        ):
            r.invoke("q")
        model.predict.assert_called_once()
        assert len(model.predict.call_args.args[0]) == 6

    def test_pairs_are_query_document_tuples(self, retrieval_settings):
        model = MagicMock()
        model.predict.return_value = [0.5] * 2
        r = CrossEncoderRetriever(k=2)
        with (
            patch.object(r, "_get_candidates", return_value=docs(2)),
            patch("src.retrieval.cross_encoder_rerank._get_cross_encoder", return_value=model),
        ):
            r.invoke("my question")
        assert model.predict.call_args.args[0][0] == ("my question", "content 0")

    def test_no_candidates_short_circuits_before_loading_the_model(self, retrieval_settings):
        r = CrossEncoderRetriever(k=4)
        with (
            patch.object(r, "_get_candidates", return_value=[]),
            patch("src.retrieval.cross_encoder_rerank._get_cross_encoder") as model,
        ):
            assert r.invoke("q") == []
        model.assert_not_called()

    def test_fewer_candidates_than_k_returns_what_exists(self, retrieval_settings):
        model = MagicMock()
        model.predict.return_value = [0.5, 0.9]
        r = CrossEncoderRetriever(k=10)
        with (
            patch.object(r, "_get_candidates", return_value=docs(2)),
            patch("src.retrieval.cross_encoder_rerank._get_cross_encoder", return_value=model),
        ):
            assert len(r.invoke("q")) == 2

    def test_candidates_come_from_the_dense_store_with_the_filter(self, retrieval_settings):
        r = CrossEncoderRetriever(k=4, fetch_k=12, filter={"department": "hr"})
        dense = MagicMock()
        dense.return_value.invoke.return_value = docs(2)
        with patch("src.vectorstore.chroma_store.get_retriever", dense):
            r._get_candidates("q")
        dense.assert_called_once_with(k=12, filter={"department": "hr"})

    def test_builder_over_fetches_relative_to_k(self, retrieval_settings):
        """Reranking needs a wider candidate pool than it returns."""
        r = build_cross_encoder_retriever(k=4)
        assert r.k == 4
        assert r.fetch_k > r.k

    def test_builder_defaults_k_to_settings(self, retrieval_settings):
        assert build_cross_encoder_retriever().k == 4

    def test_builder_caps_fetch_k(self, retrieval_settings):
        assert build_cross_encoder_retriever(k=50).fetch_k == 15


# ---------------------------------------------------------------------------
# ComposedRetriever
# ---------------------------------------------------------------------------

class TestComposedFirstStage:
    def test_first_stage_is_built_lazily_and_reused(self, retrieval_settings):
        r = ComposedRetriever(k=4, first_stage_strategy="hybrid")
        with patch("src.retrieval.factory.get_retriever") as factory:
            factory.return_value = MagicMock()
            assert r._get_first_stage() is r._get_first_stage()
        factory.assert_called_once()

    def test_first_stage_over_fetches_for_reranking(self, retrieval_settings):
        r = ComposedRetriever(k=4, first_stage_strategy="hybrid")
        with patch("src.retrieval.factory.get_retriever") as factory:
            r._get_first_stage()
        assert factory.call_args.kwargs["k"] == 12  # k * 3

    def test_first_stage_fetch_is_capped_at_15(self, retrieval_settings):
        r = ComposedRetriever(k=10)
        with patch("src.retrieval.factory.get_retriever") as factory:
            r._get_first_stage()
        assert factory.call_args.kwargs["k"] == 15

    def test_strategy_and_filter_reach_the_factory(self, retrieval_settings):
        r = ComposedRetriever(k=4, first_stage_strategy="dense", filter={"department": "legal"})
        with patch("src.retrieval.factory.get_retriever") as factory:
            r._get_first_stage()
        assert factory.call_args.kwargs["strategy"] == "dense"
        assert factory.call_args.kwargs["filter"] == {"department": "legal"}


class TestComposedLlmRerank:
    def _llm(self, scores):
        """ChatOpenAI stub whose scorer returns the given scores in order."""
        calls = {"n": 0}
        scorer = MagicMock()

        def _invoke(_messages):
            result = MagicMock()
            result.score = scores[calls["n"] % len(scores)]
            calls["n"] += 1
            return result

        scorer.invoke.side_effect = _invoke
        llm = MagicMock()
        llm.with_structured_output.return_value = scorer
        return llm

    def test_returns_top_k_by_llm_score(self, retrieval_settings):
        r = ComposedRetriever(k=2, reranker_type="llm")
        candidates = docs(3)
        with patch("src.retrieval.composed.get_llm", return_value=self._llm([1, 9, 5])):
            result = r._rerank_with_llm("q", candidates)
        assert len(result) == 2
        assert all(d in candidates for d in result)

    def test_scoring_failure_falls_back_to_a_neutral_score(self, retrieval_settings):
        """A doc whose scoring call fails must stay a candidate, not vanish."""
        scorer = MagicMock()
        scorer.invoke.side_effect = RuntimeError("rate limited")
        llm = MagicMock()
        llm.with_structured_output.return_value = scorer

        r = ComposedRetriever(k=3, reranker_type="llm")
        with patch("src.retrieval.composed.get_llm", return_value=llm):
            result = r._rerank_with_llm("q", docs(3))
        assert len(result) == 3

    def test_all_candidates_are_scored(self, retrieval_settings):
        r = ComposedRetriever(k=2, reranker_type="llm")
        llm = self._llm([5])
        with patch("src.retrieval.composed.get_llm", return_value=llm):
            r._rerank_with_llm("q", docs(5))
        assert llm.with_structured_output.return_value.invoke.call_count == 5


class TestComposedCrossEncoderRerank:
    def test_returns_top_k_by_cross_encoder_score(self, retrieval_settings):
        model = MagicMock()
        model.predict.return_value = [0.2, 0.95, 0.4]
        r = ComposedRetriever(k=1, reranker_type="cross_encoder")
        with patch(
            "src.retrieval.cross_encoder_rerank._get_cross_encoder", return_value=model
        ):
            result = r._rerank_with_cross_encoder("q", docs(3))
        assert [d.page_content for d in result] == ["content 1"]

    def test_uses_the_configured_batch_size(self, retrieval_settings):
        model = MagicMock()
        model.predict.return_value = [0.5] * 3
        r = ComposedRetriever(k=2, reranker_type="cross_encoder")
        with patch(
            "src.retrieval.cross_encoder_rerank._get_cross_encoder", return_value=model
        ):
            r._rerank_with_cross_encoder("q", docs(3))
        assert model.predict.call_args.kwargs["batch_size"] == 16


class TestComposedPipeline:
    def test_llm_path_is_taken_by_default(self, retrieval_settings):
        r = ComposedRetriever(k=2, reranker_type="llm")
        stage = MagicMock()
        stage.invoke.return_value = docs(3)
        with (
            patch.object(r, "_get_first_stage", return_value=stage),
            patch.object(r, "_rerank_with_llm", return_value=docs(2)) as llm_rerank,
            patch.object(r, "_rerank_with_cross_encoder") as ce_rerank,
        ):
            r.invoke("q")
        llm_rerank.assert_called_once()
        ce_rerank.assert_not_called()

    def test_cross_encoder_path_is_selected_by_type(self, retrieval_settings):
        r = ComposedRetriever(k=2, reranker_type="cross_encoder")
        stage = MagicMock()
        stage.invoke.return_value = docs(3)
        with (
            patch.object(r, "_get_first_stage", return_value=stage),
            patch.object(r, "_rerank_with_cross_encoder", return_value=docs(2)) as ce_rerank,
            patch.object(r, "_rerank_with_llm") as llm_rerank,
        ):
            r.invoke("q")
        ce_rerank.assert_called_once()
        llm_rerank.assert_not_called()

    def test_empty_first_stage_skips_reranking_entirely(self, retrieval_settings):
        r = ComposedRetriever(k=2)
        stage = MagicMock()
        stage.invoke.return_value = []
        with (
            patch.object(r, "_get_first_stage", return_value=stage),
            patch.object(r, "_rerank_with_llm") as rerank,
        ):
            assert r.invoke("q") == []
        rerank.assert_not_called()

    def test_query_is_passed_to_the_first_stage(self, retrieval_settings):
        r = ComposedRetriever(k=2)
        stage = MagicMock()
        stage.invoke.return_value = docs(2)
        with (
            patch.object(r, "_get_first_stage", return_value=stage),
            patch.object(r, "_rerank_with_llm", return_value=docs(2)),
        ):
            r.invoke("what is the pto policy")
        stage.invoke.assert_called_once_with("what is the pto policy")


class TestComposedBuilder:
    def test_defaults(self, retrieval_settings):
        r = build_composed_retriever()
        assert r.k == 4
        assert r.first_stage_strategy == "hybrid"
        assert r.reranker_type == "llm"

    def test_overrides_are_applied(self, retrieval_settings):
        r = build_composed_retriever(
            k=8, filter={"department": "hr"}, first_stage="dense",
            reranker_type="cross_encoder",
        )
        assert (r.k, r.first_stage_strategy, r.reranker_type) == (8, "dense", "cross_encoder")
        assert r.filter == {"department": "hr"}
