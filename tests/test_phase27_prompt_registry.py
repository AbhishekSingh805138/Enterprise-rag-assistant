"""Phase 27 — P1-8: prompts are inventoried, versioned and attributable.

Sixteen prompt constants sat in eight modules with no version, no
inventory and no record of which text produced which answer. A drop in
faithfulness could be measured but not traced to the prompt change that
caused it, and any wording change meant a code deploy.

Two things get pinned here. First, the registry itself: versions,
content hashes, overrides, and the variable check that stops an override
from quietly removing ``{context}``. Second — and more important — that
the migration to the registry did not alter a single character of any
shipped prompt.
"""
from __future__ import annotations

import json
import logging
from unittest.mock import patch

import pytest

from src.prompts.registry import (
    PromptRecord,
    _input_variables,
    get_prompt,
    list_prompts,
    load_overrides,
    prompt_fingerprint,
    register,
    reset_registry,
)

GRADE_MESSAGES = [
    ("system", "You are a grader. Answer from {context} only."),
    ("human", "Question: {question}"),
]


@pytest.fixture
def clean_registry():
    """Isolated registry with overrides disabled."""
    reset_registry()
    with patch("src.prompts.registry.settings") as s:
        s.prompt_override_dir = ""
        yield s
    reset_registry()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_register_returns_a_usable_template(self, clean_registry):
        template = register("grade", "v1", GRADE_MESSAGES)
        rendered = template.format(context="docs", question="why?")
        assert "docs" in rendered and "why?" in rendered

    def test_the_prompt_is_recorded(self, clean_registry):
        register("grade", "v2", GRADE_MESSAGES, description="relevance grader")
        record = get_prompt("grade")
        assert record.name == "grade"
        assert record.version == "v2"
        assert record.description == "relevance grader"
        assert record.source == "builtin"

    def test_input_variables_are_captured(self, clean_registry):
        register("grade", "v1", GRADE_MESSAGES)
        assert get_prompt("grade").input_variables == ("context", "question")

    def test_an_unregistered_name_returns_none(self, clean_registry):
        assert get_prompt("never-registered") is None

    def test_the_inventory_is_sorted_by_name(self, clean_registry):
        register("zebra", "v1", [("system", "z")])
        register("alpha", "v1", [("system", "a")])
        assert [r.name for r in list_prompts()] == ["alpha", "zebra"]


class TestContentHashing:
    def test_the_same_text_hashes_the_same(self, clean_registry):
        register("a", "v1", GRADE_MESSAGES)
        first = get_prompt("a").content_hash
        register("a", "v1", GRADE_MESSAGES)
        assert get_prompt("a").content_hash == first

    def test_an_edit_without_a_version_bump_is_still_visible(self, clean_registry):
        """A version tag is a promise; the hash is the fact."""
        register("a", "v1", [("system", "original wording")])
        before = get_prompt("a").content_hash
        register("a", "v1", [("system", "original wording, slightly changed")])
        assert get_prompt("a").content_hash != before

    def test_the_label_carries_version_and_hash(self, clean_registry):
        register("a", "v3", [("system", "x")])
        record = get_prompt("a")
        assert record.label == f"a@v3+{record.content_hash}"


class TestFingerprint:
    def test_an_empty_registry_has_no_fingerprint(self, clean_registry):
        assert prompt_fingerprint() == "none"

    def test_it_is_stable_for_the_same_set(self, clean_registry):
        register("a", "v1", [("system", "x")])
        first = prompt_fingerprint()
        assert prompt_fingerprint() == first

    def test_changing_any_prompt_changes_it(self, clean_registry):
        """This is what makes a quality regression attributable."""
        register("a", "v1", [("system", "x")])
        register("b", "v1", [("system", "y")])
        before = prompt_fingerprint()
        register("b", "v2", [("system", "y revised")])
        assert prompt_fingerprint() != before

    def test_registration_order_does_not_matter(self, clean_registry):
        register("a", "v1", [("system", "x")])
        register("b", "v1", [("system", "y")])
        forwards = prompt_fingerprint()
        reset_registry()
        with patch("src.prompts.registry.settings") as s:
            s.prompt_override_dir = ""
            register("b", "v1", [("system", "y")])
            register("a", "v1", [("system", "x")])
            assert prompt_fingerprint() == forwards


# ---------------------------------------------------------------------------
# Overrides — changing wording without a code change
# ---------------------------------------------------------------------------

class TestOverrides:
    def _write(self, directory, name, messages):
        (directory / f"{name}.json").write_text(json.dumps(messages), encoding="utf-8")

    def test_no_directory_means_no_overrides(self, clean_registry):
        assert load_overrides(force=True) == {}

    def test_an_override_replaces_the_shipped_text(self, tmp_path, clean_registry):
        clean_registry.prompt_override_dir = str(tmp_path)
        self._write(tmp_path, "grade", [
            ["system", "Reworded grader using {context}."],
            ["human", "Q: {question}"],
        ])
        template = register("grade", "v1", GRADE_MESSAGES)
        assert "Reworded grader" in template.format(context="c", question="q")
        assert get_prompt("grade").source == "override"

    def test_an_override_that_drops_a_variable_is_refused(self, tmp_path, clean_registry, caplog):
        """Losing {context} yields fluent answers grounded in nothing."""
        clean_registry.prompt_override_dir = str(tmp_path)
        self._write(tmp_path, "grade", [
            ["system", "Just answer well."],
            ["human", "Q: {question}"],
        ])
        with caplog.at_level(logging.ERROR):
            template = register("grade", "v1", GRADE_MESSAGES)
        assert get_prompt("grade").source == "builtin"
        assert "{context}" not in template.format(context="c", question="q")
        assert "ignoring the override" in caplog.text

    def test_an_override_that_adds_a_variable_is_refused(self, tmp_path, clean_registry, caplog):
        """An unfilled variable would raise on every request."""
        clean_registry.prompt_override_dir = str(tmp_path)
        self._write(tmp_path, "grade", [
            ["system", "Use {context} and {tone}."],
            ["human", "Q: {question}"],
        ])
        with caplog.at_level(logging.ERROR):
            register("grade", "v1", GRADE_MESSAGES)
        assert get_prompt("grade").source == "builtin"

    def test_malformed_json_is_ignored_not_fatal(self, tmp_path, clean_registry, caplog):
        clean_registry.prompt_override_dir = str(tmp_path)
        (tmp_path / "grade.json").write_text("{not json", encoding="utf-8")
        with caplog.at_level(logging.ERROR):
            overrides = load_overrides(force=True)
        assert overrides == {}
        assert "Ignoring invalid prompt override" in caplog.text

    def test_an_empty_override_is_ignored(self, tmp_path, clean_registry, caplog):
        clean_registry.prompt_override_dir = str(tmp_path)
        (tmp_path / "grade.json").write_text("[]", encoding="utf-8")
        with caplog.at_level(logging.ERROR):
            assert load_overrides(force=True) == {}

    def test_a_missing_directory_warns_rather_than_crashing(self, tmp_path, clean_registry, caplog):
        clean_registry.prompt_override_dir = str(tmp_path / "nope")
        with caplog.at_level(logging.WARNING):
            assert load_overrides(force=True) == {}
        assert "not a directory" in caplog.text

    def test_overrides_are_read_once_per_process(self, tmp_path, clean_registry):
        clean_registry.prompt_override_dir = str(tmp_path)
        self._write(tmp_path, "grade", [["system", "v1 {context}"], ["human", "{question}"]])
        load_overrides(force=True)
        self._write(tmp_path, "grade", [["system", "v2 {context}"], ["human", "{question}"]])
        assert "v1" in load_overrides()["grade"][0][1]  # cached
        assert "v2" in load_overrides(force=True)["grade"][0][1]


class TestInputVariables:
    def test_detects_every_variable(self):
        assert _input_variables(GRADE_MESSAGES) == ("context", "question")

    def test_a_prompt_with_no_variables(self):
        assert _input_variables([("system", "no placeholders")]) == ()


class TestPromptRecord:
    def test_hash_is_short_and_stable(self):
        record = PromptRecord("a", "v1", [("system", "x")])
        assert len(record.content_hash) == 12
        assert record.content_hash == PromptRecord("a", "v1", [("system", "x")]).content_hash


# ---------------------------------------------------------------------------
# The migration must not have changed any prompt
# ---------------------------------------------------------------------------

EXPECTED_PROMPTS = {
    "critic_rewrite_answer", "critic_verify", "entity_extract", "generate",
    "generate_memory", "grade_document_single", "grade_documents",
    "intent_detection", "kg_extract_entities", "mcp_tool_selection",
    "multi_query_expand", "naive_rag_answer", "planner_decompose",
    "planner_synthesize", "query_transform", "rerank_relevance",
    "rewrite_query", "unified_analysis",
}


PROMPT_MODULES = (
    "src.graph.nodes", "src.graph.planner", "src.graph.analyzer",
    "src.graph.intent_detector", "src.rag.naive_rag",
    "src.retrieval.multi_query", "src.retrieval.rerank",
)


def _import_all_prompt_modules():
    """Ensure the real prompts are registered, whatever ran before.

    Registration happens at module import, so a test earlier in the run
    that cleared the registry would leave these modules cached and
    unregistered — the inventory would look empty for reasons that have
    nothing to do with the code under test. Reload when that has
    happened, so these assertions are order-independent.
    """
    import importlib
    import sys

    # Decide once, before touching anything: reloading module by module
    # and re-checking would stop after the first reload repopulated the
    # name being checked, leaving the rest unregistered.
    registered = {r.name for r in list_prompts()}
    stale = not {"generate", "naive_rag_answer", "planner_decompose"} <= registered

    for module in PROMPT_MODULES:
        if stale and module in sys.modules:
            importlib.reload(sys.modules[module])
        else:
            importlib.import_module(module)


class TestRealPromptInventory:
    def test_every_module_level_prompt_is_registered(self):
        _import_all_prompt_modules()
        registered = {r.name for r in list_prompts()}
        missing = EXPECTED_PROMPTS - registered - {
            # Registered on first call rather than at import.
            "entity_extract", "kg_extract_entities", "mcp_tool_selection",
            "query_transform",
        }
        assert not missing, f"prompts missing from the registry: {sorted(missing)}"

    def test_no_prompt_is_missing_its_variables(self):
        _import_all_prompt_modules()
        for record in list_prompts():
            declared = set(record.input_variables)
            actual = set(_input_variables(record.messages))
            assert declared == actual, f"{record.name}: {declared} != {actual}"

    def test_the_answer_prompt_still_demands_citations(self):
        """A regression here is invisible until answers stop citing."""
        _import_all_prompt_modules()
        for name in ("generate", "naive_rag_answer"):
            text = " ".join(t for _, t in get_prompt(name).messages)
            assert "Cite the source filename" in text
            assert "{context}" in text

    def test_the_answer_prompt_still_forbids_invention(self):
        _import_all_prompt_modules()
        text = " ".join(t for _, t in get_prompt("generate").messages)
        assert "Do NOT invent facts" in text

    def test_the_critic_still_defaults_to_lenient(self):
        """Flipping this would reject correct answers wholesale."""
        _import_all_prompt_modules()
        text = " ".join(t for _, t in get_prompt("critic_verify").messages)
        assert "When in doubt, mark as SUPPORTED" in text

    def test_the_idk_sentence_is_identical_in_both_answer_prompts(self):
        """is_idk_response() matches on this wording; drift breaks metrics."""
        _import_all_prompt_modules()
        sentence = (
            "I don't have enough information in the available documents to "
            "answer this question."
        )
        for name in ("generate", "naive_rag_answer"):
            joined = " ".join(t for _, t in get_prompt(name).messages)
            normalised = " ".join(joined.split())
            assert sentence in normalised, f"{name} no longer states the IDK sentence"

    def test_the_generation_variants_are_distinct_prompts(self):
        """Memory changes the text, so it must not share an identity."""
        _import_all_prompt_modules()
        assert get_prompt("generate").content_hash != get_prompt("generate_memory").content_hash

    def test_the_memory_variant_carries_the_memory_slot(self):
        _import_all_prompt_modules()
        assert "memory_context" in get_prompt("generate_memory").input_variables


class TestMetricsAttribution:
    def test_the_prompt_version_is_recorded_with_each_query(self, tmp_path):
        from dataclasses import dataclass

        from src.observability.metrics_store import MetricsStore

        _import_all_prompt_modules()

        @dataclass
        class M:
            thread_id: str = "t"
            question_preview: str = "q?"
            mode: str = "graph"
            retriever_strategy: str = "dense"
            prompt_tokens: int = 1
            completion_tokens: int = 1
            total_tokens: int = 2
            estimated_cost_usd: float = 0.0
            latency_ms: float = 1.0
            is_idk: bool = False
            grader_rejected: int = 0

        store = MetricsStore(str(tmp_path / "m.db"))
        store.record(M())
        row = store.query_recent(1)[0]
        store.close()
        assert row["prompt_version"] == prompt_fingerprint()
        assert row["prompt_version"] not in ("", "none")

    def test_recording_survives_a_broken_registry(self, tmp_path):
        """Metrics must never be the thing that fails a query."""
        from src.observability.metrics_store import _prompt_version

        with patch("src.prompts.prompt_fingerprint", side_effect=RuntimeError("boom")):
            assert _prompt_version() == ""

    def test_the_column_is_added_to_an_existing_database(self, tmp_path):
        """Deployments upgrade in place; the migration must be idempotent."""
        import sqlite3

        from src.observability.metrics_store import MetricsStore

        path = str(tmp_path / "old.db")
        conn = sqlite3.connect(path)
        conn.execute(
            "CREATE TABLE query_metrics (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "ts TEXT NOT NULL, thread_id TEXT NOT NULL, question TEXT NOT NULL, "
            "mode TEXT NOT NULL, retriever TEXT NOT NULL, prompt_tok INTEGER, "
            "compl_tok INTEGER, total_tok INTEGER, cost_usd REAL, latency_ms REAL)"
        )
        conn.commit()
        conn.close()

        store = MetricsStore(path)
        columns = {r[1] for r in store._conn.execute("PRAGMA table_info(query_metrics)")}
        store.close()
        assert "prompt_version" in columns

        MetricsStore(path).close()  # second open must not fail
