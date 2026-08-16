"""Phase 27 — P1-2: the LLM vendor is a configuration choice, with failover.

``ChatOpenAI`` and ``OpenAIEmbeddings`` were constructed directly in nine
modules, so an OpenAI outage was a total outage — the circuit breakers
failed fast but there was nothing to fail over to.

Chat models and embeddings are treated differently on purpose. Chat
models are interchangeable, so a fallback vendor is real failover.
Embedding models are not: their vectors are not comparable, so switching
one silently returns confident nonsense. That asymmetry is what most of
this file pins down.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from src.llm.providers import (
    ChatModelWithFallbacks,
    ProviderNotAvailable,
    build_chat_model,
    build_embeddings,
    embedding_fingerprint,
    fallback_spec,
)


class Verdict(BaseModel):
    ok: bool


class _FlakyModel(Runnable):
    """Stands in for a vendor that is down."""

    def __init__(self, error: Exception):
        self.error = error

    def invoke(self, input, config=None, **kwargs):
        raise self.error

    def with_structured_output(self, schema, **kwargs):
        return self

    def bind_tools(self, tools, **kwargs):
        return self


class _WorkingModel(Runnable):
    def __init__(self, reply="from fallback"):
        self.reply = reply
        self.calls = 0

    def invoke(self, input, config=None, **kwargs):
        self.calls += 1
        return AIMessage(content=self.reply)

    def with_structured_output(self, schema, **kwargs):
        model = _WorkingModel(self.reply)
        model.invoke = lambda i, c=None, **k: schema(ok=True)  # type: ignore[assignment]
        return model

    def bind_tools(self, tools, **kwargs):
        return self


# ---------------------------------------------------------------------------
# Provider construction
# ---------------------------------------------------------------------------

class TestBuildChatModel:
    def test_openai_is_the_default_path(self):
        with patch("src.llm.providers.settings") as s:
            s.openai_api_key = "sk-test"
            s.llm_timeout = 30
            s.llm_max_retries = 2
            model = build_chat_model("openai", "gpt-4o-mini", 0.0)
        assert type(model).__name__ == "ChatOpenAI"

    def test_settings_reach_the_client(self):
        with patch("src.llm.providers.settings") as s:
            s.openai_api_key = "sk-test"
            s.llm_timeout = 45
            s.llm_max_retries = 4
            model = build_chat_model("openai", "gpt-4o-mini", 0.3)
        assert model.request_timeout == 45
        assert model.max_retries == 4
        assert model.temperature == 0.3

    def test_unknown_provider_is_rejected_by_name(self):
        with pytest.raises(ProviderNotAvailable, match="cohere"):
            build_chat_model("cohere", "some-model")

    def test_provider_name_is_case_and_space_insensitive(self):
        with patch("src.llm.providers.settings") as s:
            s.openai_api_key = "sk-test"
            s.llm_timeout = 30
            s.llm_max_retries = 2
            assert build_chat_model("  OpenAI ", "gpt-4o-mini") is not None

    def test_azure_without_an_endpoint_fails_with_a_usable_message(self):
        with patch("src.llm.providers.settings") as s:
            s.azure_openai_endpoint = ""
            s.llm_timeout = 30
            s.llm_max_retries = 2
            with pytest.raises(ProviderNotAvailable, match="AZURE_OPENAI_ENDPOINT"):
                build_chat_model("azure_openai", "gpt-4o")

    def test_anthropic_without_a_key_fails_before_the_first_call(self):
        pytest.importorskip("langchain_anthropic")
        with patch("src.llm.providers.settings") as s:
            s.anthropic_api_key = ""
            s.llm_timeout = 30
            s.llm_max_retries = 2
            with pytest.raises(ProviderNotAvailable, match="ANTHROPIC_API_KEY"):
                build_chat_model("anthropic", "claude-sonnet-5")


# ---------------------------------------------------------------------------
# Failover
# ---------------------------------------------------------------------------

class TestChatModelWithFallbacks:
    def test_primary_is_used_when_healthy(self):
        primary, secondary = _WorkingModel("primary"), _WorkingModel("secondary")
        model = ChatModelWithFallbacks(primary, [secondary])
        assert model.invoke("hi").content == "primary"
        assert secondary.calls == 0

    def test_an_outage_falls_through_to_the_second_vendor(self):
        """The whole point: a vendor outage stops being a total outage."""
        primary = _FlakyModel(RuntimeError("service unavailable"))
        secondary = _WorkingModel("from anthropic")
        model = ChatModelWithFallbacks(primary, [secondary])
        assert model.invoke("hi").content == "from anthropic"

    def test_structured_output_survives_failover(self):
        """The trap: RunnableWithFallbacks has no with_structured_output.

        Graders, planners and the critic are all built through that
        method. Wrapping the model naively would have broken every typed
        node the moment failover was switched on.
        """
        primary = _FlakyModel(RuntimeError("down"))
        model = ChatModelWithFallbacks(primary, [_WorkingModel()])
        chain = model.with_structured_output(Verdict)
        assert chain.invoke("check this") == Verdict(ok=True)

    def test_structured_output_is_still_typed_without_a_fallback(self):
        model = ChatModelWithFallbacks(_WorkingModel(), [])
        assert model.with_structured_output(Verdict).invoke("x") == Verdict(ok=True)

    def test_a_fallback_that_cannot_do_structured_output_is_skipped(self, caplog):
        """One vendor's limitation must not break the primary path."""

        class _NoSchemas(_WorkingModel):
            def with_structured_output(self, schema, **kwargs):
                raise NotImplementedError("no schemas")

        model = ChatModelWithFallbacks(_WorkingModel(), [_NoSchemas()])
        with caplog.at_level(logging.WARNING):
            chain = model.with_structured_output(Verdict)
        assert chain.invoke("x") == Verdict(ok=True)
        assert "cannot produce structured output" in caplog.text

    def test_no_fallback_configured_leaves_the_model_alone(self):
        primary = _WorkingModel("only")
        model = ChatModelWithFallbacks(primary, [])
        assert model._chain is primary

    def test_attribute_reads_resolve_against_the_primary(self):
        primary = _WorkingModel()
        primary.model_name = "gpt-4o-mini"
        model = ChatModelWithFallbacks(primary, [_WorkingModel()])
        assert model.model_name == "gpt-4o-mini"

    def test_it_composes_into_a_chain(self):
        """`prompt | llm` must keep working."""
        from langchain_core.prompts import ChatPromptTemplate

        model = ChatModelWithFallbacks(_WorkingModel("composed"), [])
        chain = ChatPromptTemplate.from_messages([("human", "{q}")]) | model
        assert chain.invoke({"q": "hi"}).content == "composed"

    def test_repr_names_the_vendors(self):
        model = ChatModelWithFallbacks(_WorkingModel(), [_WorkingModel()])
        assert "primary=_WorkingModel" in repr(model)


class TestFallbackSpec:
    def test_disabled_by_default(self):
        with patch("src.llm.providers.settings") as s:
            s.llm_fallback_provider = ""
            assert fallback_spec() is None

    def test_model_defaults_to_the_primary_model(self):
        with patch("src.llm.providers.settings") as s:
            s.llm_fallback_provider = "anthropic"
            s.llm_fallback_model = ""
            s.llm_model = "gpt-4o-mini"
            assert fallback_spec() == ("anthropic", "gpt-4o-mini")

    def test_explicit_fallback_model_wins(self):
        with patch("src.llm.providers.settings") as s:
            s.llm_fallback_provider = "anthropic"
            s.llm_fallback_model = "claude-sonnet-5"
            assert fallback_spec() == ("anthropic", "claude-sonnet-5")


class TestPoolWiring:
    def test_without_a_fallback_the_pool_returns_the_bare_model(self):
        """Zero behaviour change for deployments that do not opt in."""
        from src.llm_pool import get_llm, reset_pool

        reset_pool()
        with patch("src.llm.providers.settings") as ps, patch("src.llm_pool.settings") as ls:
            ps.openai_api_key = "sk-test"
            ps.llm_timeout = 30
            ps.llm_max_retries = 2
            ls.llm_model = "gpt-4o-mini"
            ls.llm_provider = "openai"
            ls.llm_fallback_provider = ""
            model = get_llm()
        reset_pool()
        assert not isinstance(model, ChatModelWithFallbacks)

    def test_a_configured_fallback_is_wrapped(self):
        from src.llm_pool import get_llm, reset_pool

        reset_pool()
        with (
            patch("src.llm_pool.settings") as ls,
            patch("src.llm_pool.build_chat_model", side_effect=[_WorkingModel("p"), _WorkingModel("f")]),
            patch("src.llm_pool.fallback_spec", return_value=("anthropic", "claude-sonnet-5")),
        ):
            ls.llm_model = "gpt-4o-mini"
            ls.llm_provider = "openai"
            model = get_llm()
        reset_pool()
        assert isinstance(model, ChatModelWithFallbacks)

    def test_an_unusable_fallback_is_reported_and_does_not_break_the_primary(self, caplog):
        """Believing you have failover when you do not is worse than none."""
        from src.llm_pool import get_llm, reset_pool

        reset_pool()
        with (
            patch("src.llm_pool.settings") as ls,
            patch(
                "src.llm_pool.build_chat_model",
                side_effect=[_WorkingModel("p"), ProviderNotAvailable("no package")],
            ),
            patch("src.llm_pool.fallback_spec", return_value=("anthropic", "claude-sonnet-5")),
            caplog.at_level(logging.ERROR),
        ):
            ls.llm_model = "gpt-4o-mini"
            ls.llm_provider = "openai"
            model = get_llm()
        reset_pool()
        assert not isinstance(model, ChatModelWithFallbacks)
        assert "WITHOUT failover" in caplog.text

    def test_the_pool_still_caches(self):
        from src.llm_pool import get_llm, reset_pool

        reset_pool()
        with patch("src.llm_pool.build_chat_model", return_value=_WorkingModel()) as build:
            with patch("src.llm_pool.settings") as ls:
                ls.llm_model = "gpt-4o-mini"
                ls.llm_provider = "openai"
                ls.llm_fallback_provider = ""
                with patch("src.llm_pool.fallback_spec", return_value=None):
                    assert get_llm() is get_llm()
        reset_pool()
        assert build.call_count == 1


class TestNoDirectVendorConstruction:
    """Any module building a client directly is outside failover."""

    def test_no_source_module_instantiates_a_vendor_class(self):
        import pathlib
        import re

        root = pathlib.Path(__file__).resolve().parent.parent
        offenders = []
        for path in (root / "src").rglob("*.py"):
            if path.parts[-2:] == ("llm", "providers.py"):
                continue  # the one place that is allowed to
            text = path.read_text(encoding="utf-8")
            if re.search(r"\b(ChatOpenAI|OpenAIEmbeddings|ChatAnthropic)\s*\(", text):
                offenders.append(str(path.relative_to(root)))
        assert not offenders, (
            "these modules construct a vendor client directly and would not "
            f"fail over: {offenders}"
        )


# ---------------------------------------------------------------------------
# Embeddings are deliberately NOT interchangeable
# ---------------------------------------------------------------------------

class TestEmbeddingProviders:
    def test_openai_embeddings_build(self):
        with patch("src.llm.providers.settings") as s:
            s.openai_api_key = "sk-test"
            assert build_embeddings("openai", "text-embedding-3-small") is not None

    def test_anthropic_is_refused_with_the_reason(self):
        with pytest.raises(ProviderNotAvailable, match="no embedding API"):
            build_embeddings("anthropic", "whatever")

    def test_unknown_provider_is_refused(self):
        with pytest.raises(ProviderNotAvailable, match="Unknown embedding provider"):
            build_embeddings("cohere", "whatever")

    def test_fingerprint_identifies_the_vector_space(self):
        with patch("src.llm.providers.settings") as s:
            s.embedding_provider = "openai"
            s.embedding_model = "text-embedding-3-small"
            assert embedding_fingerprint() == "openai|text-embedding-3-small"

    def test_changing_the_model_changes_the_fingerprint(self):
        with patch("src.llm.providers.settings") as s:
            s.embedding_provider = "openai"
            s.embedding_model = "text-embedding-3-small"
            small = embedding_fingerprint()
            s.embedding_model = "text-embedding-3-large"
            assert embedding_fingerprint() != small

    def test_azure_deployment_is_part_of_the_identity(self):
        """Two deployments can serve different models under one name."""
        with patch("src.llm.providers.settings") as s:
            s.embedding_provider = "azure_openai"
            s.embedding_model = "text-embedding-3-small"
            s.azure_embedding_deployment = "deploy-a"
            a = embedding_fingerprint()
            s.azure_embedding_deployment = "deploy-b"
            assert embedding_fingerprint() != a


class TestEmbeddingSpaceGuard:
    """A changed embedding model must not silently poison retrieval."""

    def _store(self, recorded=None, count=5):
        store = MagicMock()
        store._collection.metadata = {"embedding_fingerprint": recorded} if recorded else {}
        store._collection.count.return_value = count
        return store

    def test_a_mismatch_refuses_to_serve(self):
        from src.vectorstore.chroma_store import EmbeddingSpaceMismatch, _assert_embedding_space

        with patch("src.vectorstore.chroma_store.embedding_fingerprint", return_value="openai|new"):
            with pytest.raises(EmbeddingSpaceMismatch, match="not comparable"):
                _assert_embedding_space(self._store(recorded="openai|old"))

    def test_the_error_names_both_models_and_the_way_out(self):
        from src.vectorstore.chroma_store import EmbeddingSpaceMismatch, _assert_embedding_space

        with patch("src.vectorstore.chroma_store.embedding_fingerprint", return_value="openai|new"):
            with pytest.raises(EmbeddingSpaceMismatch) as exc:
                _assert_embedding_space(self._store(recorded="openai|old"))
        message = str(exc.value)
        assert "openai|old" in message and "openai|new" in message
        assert "Re-index" in message

    def test_a_matching_fingerprint_passes(self):
        from src.vectorstore.chroma_store import _assert_embedding_space

        with patch("src.vectorstore.chroma_store.embedding_fingerprint", return_value="openai|same"):
            _assert_embedding_space(self._store(recorded="openai|same"))

    def test_an_empty_collection_is_claimed_silently(self, caplog):
        from src.vectorstore.chroma_store import _assert_embedding_space

        store = self._store(recorded=None, count=0)
        with patch("src.vectorstore.chroma_store.embedding_fingerprint", return_value="openai|x"):
            with caplog.at_level(logging.WARNING):
                _assert_embedding_space(store)
        store._collection.modify.assert_called_once()
        assert caplog.text == ""

    def test_an_existing_corpus_adopts_the_fingerprint_but_says_so(self, caplog):
        """The one case a real mismatch could slip through unnoticed."""
        from src.vectorstore.chroma_store import _assert_embedding_space

        store = self._store(recorded=None, count=114)
        with patch("src.vectorstore.chroma_store.embedding_fingerprint", return_value="openai|x"):
            with caplog.at_level(logging.WARNING):
                _assert_embedding_space(store)
        assert "no embedding fingerprint" in caplog.text
        store._collection.modify.assert_called_once()

    def test_the_guard_never_takes_down_retrieval_itself(self):
        from src.vectorstore.chroma_store import _assert_embedding_space

        store = MagicMock()
        type(store)._collection = property(lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
        _assert_embedding_space(store)  # must not raise

    def test_a_non_string_fingerprint_is_not_treated_as_one(self):
        from src.vectorstore.chroma_store import _assert_embedding_space

        store = MagicMock()
        store._collection.metadata = {"embedding_fingerprint": object()}
        store._collection.count.return_value = 0
        with patch("src.vectorstore.chroma_store.embedding_fingerprint", return_value="openai|x"):
            _assert_embedding_space(store)  # must not raise


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------

class TestProviderConfigValidation:
    def _settings(self, **over):
        import config

        base = {"openai_api_key": "sk-test"}
        base.update(over)
        return config.Settings(**base)

    def test_defaults_validate(self):
        self._settings().validate()

    def test_unknown_chat_provider_is_rejected(self):
        with pytest.raises(ValueError, match="LLM_PROVIDER"):
            self._settings(llm_provider="cohere").validate()

    def test_unknown_embedding_provider_is_rejected(self):
        with pytest.raises(ValueError, match="EMBEDDING_PROVIDER"):
            self._settings(embedding_provider="cohere").validate()

    def test_anthropic_embeddings_are_rejected(self):
        with pytest.raises(ValueError, match="EMBEDDING_PROVIDER"):
            self._settings(embedding_provider="anthropic").validate()

    def test_a_fallback_on_the_same_vendor_is_rejected(self):
        """It would look configured while surviving nothing."""
        with pytest.raises(ValueError, match="same vendor"):
            self._settings(llm_provider="openai", llm_fallback_provider="openai").validate()

    def test_a_real_fallback_is_accepted(self):
        self._settings(llm_provider="openai", llm_fallback_provider="anthropic").validate()

    def test_openai_key_is_not_required_when_openai_is_unused(self):
        self._settings(
            openai_api_key="",
            llm_provider="anthropic",
            embedding_provider="azure_openai",
        ).validate()

    def test_openai_key_is_still_required_on_the_default_path(self):
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            self._settings(openai_api_key="").validate()

    def test_openai_key_is_required_when_only_the_fallback_uses_it(self):
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            self._settings(
                openai_api_key="",
                llm_provider="anthropic",
                embedding_provider="azure_openai",
                llm_fallback_provider="openai",
            ).validate()

    def test_shipped_defaults_are_single_vendor(self):
        import inspect
        import re

        import config

        source = inspect.getsource(config.Settings)
        assert re.search(r'llm_provider:\s*str\s*=\s*os\.getenv\(\s*"LLM_PROVIDER",\s*"openai"\s*\)', source)
        assert re.search(r'llm_fallback_provider:\s*str\s*=\s*os\.getenv\(\s*"LLM_FALLBACK_PROVIDER",\s*""\s*\)', source)
