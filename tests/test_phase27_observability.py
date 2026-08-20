"""Phase 27 — P1-7: metrics leave the box, and something pages a human.

Cost, latency, IDK rate and queue depth were all measured — and all of it
landed in SQLite readable only by a CLI. That is a record, not
monitoring: nothing was graphed and nothing alerted. A non-zero
dead-letter queue degraded the health check, but only if a person looked.

Two additions and one rule of thumb they both follow: observability must
never be able to break the thing it observes. A failing collector costs
one series, not the scrape; a tracing problem costs a span, not the
request.
"""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

import src.observability.prometheus as prom

ALERTS_FILE = Path(__file__).resolve().parent.parent / "deploy" / "prometheus" / "alerts.yml"


@pytest.fixture(autouse=True)
def _clean():
    prom.reset_cache()
    yield
    prom.reset_cache()


def _store(**over):
    store = MagicMock()
    store.summary.return_value = {
        "cnt": 100, "total_cost": 0.5, "avg_cost": 0.005,
        "avg_latency": 2000.0, "total_tokens": 100_000, "over_budget": 3,
    }
    store.spend_today.return_value = 0.12
    store.latency_percentiles.return_value = {"p50": 1800.0, "p95": 6000.0, "p99": 9000.0}
    store.idk_rate.return_value = 0.15
    store.grader_rejection_rate.return_value = 0.22
    for key, value in over.items():
        getattr(store, key).return_value = value
    return store


def _bus(backlog=4, dlq=0):
    bus = MagicMock()
    bus.depth.side_effect = lambda topic: dlq if "dlq" in topic else backlog
    return bus


def _registry(**counts):
    registry = MagicMock()
    stats = {"PENDING": 0, "PROCESSING": 0, "PROCESSED": 0, "FAILED": 0, "DEAD_LETTER": 0}
    stats.update(counts)
    stats["TOTAL"] = sum(stats.values())
    registry.stats.return_value = stats
    return registry


def render(store=None, bus=None, registry=None):
    with (
        patch("src.observability.metrics_store.get_store", return_value=store or _store()),
        patch("src.events.bus.get_event_bus", return_value=bus or _bus()),
        patch("src.ingestion.registry.get_registry", return_value=registry or _registry()),
    ):
        return prom.render(force=True)


def series(body: str, name: str) -> list[str]:
    return [
        line for line in body.splitlines()
        if line.startswith(name) and not line.startswith("#")
    ]


# ---------------------------------------------------------------------------
# Exposition format
# ---------------------------------------------------------------------------

class TestExpositionFormat:
    def test_every_series_declares_help_and_type(self):
        """Without these a metric is unusable in any dashboard tool."""
        body = render()
        names = {
            line.split()[0] for line in body.splitlines()
            if line and not line.startswith("#")
        }
        bare = {n.split("{")[0] for n in names}
        for name in bare:
            assert f"# HELP {name} " in body, f"{name} has no HELP"
            assert f"# TYPE {name} " in body, f"{name} has no TYPE"

    def test_values_are_numeric(self):
        for line in render().splitlines():
            if not line or line.startswith("#"):
                continue
            float(line.rsplit(" ", 1)[1])

    def test_it_ends_with_a_newline(self):
        """Prometheus rejects a body whose last line is unterminated."""
        assert render().endswith("\n")

    def test_labels_are_quoted(self):
        body = render(registry=_registry(PROCESSED=6))
        assert 'rag_documents{status="processed"} 6' in body


class TestQueryMetrics:
    def test_cost_and_volume_are_exported(self):
        body = render()
        assert "rag_queries_total 100" in body
        assert "rag_cost_usd_total 0.5" in body
        assert "rag_cost_usd_today 0.12" in body

    def test_latency_percentiles_are_exported(self):
        body = render()
        assert 'rag_query_latency_ms{quantile="0.95"} 6000.0' in body

    def test_quality_rates_are_exported(self):
        """The RAG-specific ones — the reason this system needs alerts."""
        body = render()
        assert "rag_idk_rate 0.15" in body
        assert "rag_grader_rejection_rate 0.22" in body

    def test_rates_use_a_recent_window_not_all_time(self):
        """An all-time rate never recovers, so an alert on it is useless."""
        store = _store()
        render(store=store)
        store.idk_rate.assert_called_with(prom.RECENT_WINDOW)
        store.grader_rejection_rate.assert_called_with(prom.RECENT_WINDOW)

    def test_the_daily_cap_is_exported_for_alerting(self):
        """So an alert can fire before the cap starts returning 503s."""
        assert "rag_cost_daily_cap_usd" in render()


class TestIngestionMetrics:
    def test_queue_and_dlq_depth_are_separate_series(self):
        body = render(bus=_bus(backlog=17, dlq=2))
        assert "rag_ingestion_queue_depth 17" in body
        assert "rag_ingestion_dlq_depth 2" in body

    def test_the_queue_limit_is_exported(self):
        """An alert needs the threshold, not just the value."""
        assert series(render(), "rag_ingestion_queue_limit")

    def test_documents_are_broken_down_by_status(self):
        body = render(registry=_registry(PROCESSED=100, DEAD_LETTER=3))
        assert 'rag_documents{status="processed"} 100' in body
        assert 'rag_documents{status="dead_letter"} 3' in body

    def test_the_total_is_not_exported_as_a_status(self):
        """Prometheus sums the series; a TOTAL label would double-count."""
        assert 'status="total"' not in render()


class TestBuildInfo:
    def test_prompt_set_and_models_are_labels(self):
        body = render()
        line = series(body, "rag_build_info")[0]
        assert "prompt_set=" in line
        assert "llm_model=" in line
        assert line.endswith(" 1")


# ---------------------------------------------------------------------------
# The rule that matters most
# ---------------------------------------------------------------------------

class TestCollectorIsolation:
    def test_one_failing_collector_does_not_fail_the_scrape(self, caplog):
        """A 500 here takes monitoring down exactly when it is needed."""
        with (
            patch("src.observability.metrics_store.get_store", side_effect=RuntimeError("db gone")),
            patch("src.events.bus.get_event_bus", return_value=_bus()),
            patch("src.ingestion.registry.get_registry", return_value=_registry()),
            caplog.at_level(logging.WARNING),
        ):
            body = prom.render(force=True)

        assert "rag_ingestion_queue_depth" in body  # other collectors survived
        assert "rag_metrics_collector_errors 1" in body
        assert "collector 'query' failed" in caplog.text.replace('"', "'")

    def test_a_failure_is_visible_rather_than_looking_like_zero(self):
        """Missing and zero are different, and only one needs a human."""
        with (
            patch("src.observability.metrics_store.get_store", side_effect=RuntimeError("x")),
            patch("src.events.bus.get_event_bus", side_effect=RuntimeError("y")),
        ):
            body = prom.render(force=True)
        assert "rag_metrics_collector_errors 2" in body
        assert "rag_idk_rate" not in body  # absent, not reported as 0

    def test_all_collectors_healthy_reports_zero_errors(self):
        assert "rag_metrics_collector_errors 0" in render()


class TestScrapeCost:
    def test_the_body_is_cached_between_scrapes(self):
        """Several of these aggregates are full-table scans."""
        store = _store()
        with (
            patch("src.observability.metrics_store.get_store", return_value=store),
            patch("src.events.bus.get_event_bus", return_value=_bus()),
            patch("src.ingestion.registry.get_registry", return_value=_registry()),
        ):
            prom.render(force=True)
            prom.render()
            prom.render()
        assert store.summary.call_count == 2  # one full + one recent, once

    def test_force_bypasses_the_cache(self):
        store = _store()
        with (
            patch("src.observability.metrics_store.get_store", return_value=store),
            patch("src.events.bus.get_event_bus", return_value=_bus()),
            patch("src.ingestion.registry.get_registry", return_value=_registry()),
        ):
            prom.render(force=True)
            prom.render(force=True)
        assert store.summary.call_count == 4


# ---------------------------------------------------------------------------
# The endpoint
# ---------------------------------------------------------------------------

class TestMetricsEndpoint:
    def _client(self, auth_enabled):
        from fastapi.testclient import TestClient

        s = MagicMock()
        s.validate = MagicMock()
        s.auth_enabled = auth_enabled
        s.api_keys = "metrics-key-aaaaaaaaaaaa"
        s.chroma_collection = "c"
        s.log_level = "WARNING"
        s.debug_mode = False
        s.cors_origins = "http://localhost:8501"
        s.cors_allow_methods = "GET,POST,OPTIONS"
        s.cors_allow_headers = "Content-Type"
        s.async_ingestion = False
        s.rate_limit_storage_uri = ""
        return s

    @pytest.fixture
    def secured(self):
        from fastapi.testclient import TestClient

        s = self._client(auth_enabled=True)
        with patch("api.app.settings", s), patch("src.security.auth.settings", s):
            from api.app import app

            with TestClient(app) as tc:
                yield tc

    @pytest.fixture
    def open_api(self):
        from fastapi.testclient import TestClient

        s = self._client(auth_enabled=False)
        with patch("api.app.settings", s), patch("src.security.auth.settings", s):
            from api.app import app

            with TestClient(app) as tc:
                yield tc

    def test_metrics_requires_a_key(self, secured):
        """The body carries queue depths and per-department document counts."""
        with patch("api.app.settings"):
            resp = secured.get("/metrics")
        assert resp.status_code == 401

    def test_a_valid_key_gets_the_exposition(self, secured):
        with patch("src.observability.prometheus.render", return_value="rag_up 1\n"):
            resp = secured.get(
                "/metrics", headers={"Authorization": "Bearer metrics-key-aaaaaaaaaaaa"}
            )
        assert resp.status_code == 200
        assert resp.text == "rag_up 1\n"

    def test_the_content_type_is_scrapeable(self, secured):
        """JSON here would simply not be parsed by Prometheus."""
        with patch("src.observability.prometheus.render", return_value="rag_up 1\n"):
            resp = secured.get(
                "/metrics", headers={"Authorization": "Bearer metrics-key-aaaaaaaaaaaa"}
            )
        assert resp.headers["content-type"].startswith("text/plain")

    def test_it_is_open_when_auth_is_disabled(self, open_api):
        with patch("src.observability.prometheus.render", return_value="rag_up 1\n"):
            assert open_api.get("/metrics").status_code == 200


# ---------------------------------------------------------------------------
# Alert rules
# ---------------------------------------------------------------------------

class TestAlertRules:
    @pytest.fixture(scope="class")
    def rules(self):
        data = yaml.safe_load(ALERTS_FILE.read_text(encoding="utf-8"))
        return [r for group in data["groups"] for r in group["rules"]]

    def test_the_file_is_valid_yaml_with_groups(self, rules):
        assert rules

    def test_every_rule_is_complete(self, rules):
        for rule in rules:
            assert rule.get("alert"), rule
            assert rule.get("expr"), rule
            assert rule.get("annotations", {}).get("summary"), rule["alert"]

    def test_every_rule_has_a_severity(self, rules):
        for rule in rules:
            assert rule["labels"]["severity"] in {"page", "ticket"}, rule["alert"]

    def test_every_rule_waits_before_firing(self, rules):
        """Without `for`, a single scrape blip pages someone at 3am."""
        for rule in rules:
            assert rule.get("for"), f"{rule['alert']} has no `for` duration"

    @pytest.mark.parametrize(
        "alert",
        [
            "RagDeadLetterQueueNonEmpty",   # the review's headline alert
            "RagIdkRateSpike",              # RAG-specific quality collapse
            "RagP95LatencyHigh",
            "RagDailySpendApproachingCap",
        ],
    )
    def test_the_four_named_conditions_are_covered(self, rules, alert):
        assert alert in {r["alert"] for r in rules}

    def test_every_referenced_series_is_actually_exported(self, rules):
        """An alert on a series nobody exports never fires, and looks fine."""
        import re

        exported = {
            line.split()[2] for line in render().splitlines()
            if line.startswith("# TYPE ")
        }
        exported |= {"up"}  # Prometheus's own target-health series

        referenced = set()
        for rule in rules:
            referenced |= set(re.findall(r"\brag_[a-z_]+\b", rule["expr"]))
            referenced |= set(re.findall(r"\bup\b", rule["expr"]))

        missing = referenced - exported
        assert not missing, f"alerts reference unexported series: {sorted(missing)}"

    def test_the_dead_letter_alert_pages_rather_than_ticketing(self, rules):
        """These documents are never indexed without a human."""
        rule = next(r for r in rules if r["alert"] == "RagDeadLetterQueueNonEmpty")
        assert rule["labels"]["severity"] == "page"
        assert "replay_dlq" in rule["annotations"]["description"]

    def test_backlog_alerts_on_the_trend_not_the_depth(self, rules):
        """A burst upload is legitimately deep and drains fine."""
        rule = next(r for r in rules if r["alert"] == "RagIngestionBacklogNotDraining")
        assert "deriv(" in rule["expr"]


# ---------------------------------------------------------------------------
# Tracing across the queue
# ---------------------------------------------------------------------------

class TestOtelTracing:
    @pytest.fixture(autouse=True)
    def _reset(self):
        from src.observability import tracing_otel as otel

        otel.reset_for_tests()
        yield
        otel.reset_for_tests()

    def test_disabled_by_default(self):
        from src.observability import tracing_otel as otel

        with patch("src.observability.tracing_otel.settings") as s:
            s.otel_enabled = False
            assert otel.setup_tracing() is False

    def test_spans_are_no_ops_when_disabled(self):
        from src.observability import tracing_otel as otel

        with patch("src.observability.tracing_otel.settings") as s:
            s.otel_enabled = False
            with otel.span("anything", **{"document.id": "d1"}) as current:
                assert current is None

    def test_the_payload_is_untouched_when_disabled(self):
        """A non-traced deployment's envelope gains no extra field."""
        from src.observability import tracing_otel as otel

        with patch("src.observability.tracing_otel.settings") as s:
            s.otel_enabled = False
            payload = {"storage_key": "k"}
            assert otel.inject_context(payload) == payload

    def test_the_context_is_injected_when_enabled(self):
        pytest.importorskip("opentelemetry.sdk")
        from src.observability import tracing_otel as otel

        with patch("src.observability.tracing_otel.settings") as s:
            s.otel_enabled = True
            s.otel_service_name = "test"
            s.otel_service_version = "1.0"
            s.otel_exporter_endpoint = ""
            with otel.span("ingestion.publish"):
                payload = otel.inject_context({"storage_key": "k"})
        assert otel.TRACE_CONTEXT_KEY in payload
        assert payload["storage_key"] == "k"

    def test_a_worker_rejoins_the_publishing_trace(self):
        """The whole point: one trace across two processes and a queue."""
        pytest.importorskip("opentelemetry.sdk")
        from opentelemetry import trace

        from src.observability import tracing_otel as otel

        with patch("src.observability.tracing_otel.settings") as s:
            s.otel_enabled = True
            s.otel_service_name = "test"
            s.otel_service_version = "1.0"
            s.otel_exporter_endpoint = ""

            with otel.span("ingestion.publish") as publish_span:
                published_trace = publish_span.get_span_context().trace_id
                payload = otel.inject_context({"storage_key": "k"})

            # ...minutes later, in the worker process:
            with otel.consumer_span("ingestion.index", payload) as index_span:
                assert index_span.get_span_context().trace_id == published_trace
        assert published_trace != trace.INVALID_TRACE_ID

    def test_an_event_without_a_context_still_gets_its_own_span(self):
        """Legacy events predate the field; they must still be traceable."""
        pytest.importorskip("opentelemetry.sdk")
        from src.observability import tracing_otel as otel

        with patch("src.observability.tracing_otel.settings") as s:
            s.otel_enabled = True
            s.otel_service_name = "test"
            s.otel_service_version = "1.0"
            s.otel_exporter_endpoint = ""
            with otel.consumer_span("ingestion.index", {"storage_key": "k"}) as current:
                assert current is not None

    def test_a_corrupt_context_does_not_raise(self):
        from src.observability import tracing_otel as otel

        with patch("src.observability.tracing_otel.settings") as s:
            s.otel_enabled = True
            s.otel_service_name = "test"
            s.otel_service_version = "1.0"
            s.otel_exporter_endpoint = ""
            assert otel.extract_context({otel.TRACE_CONTEXT_KEY: "not-a-dict"}) is None

    def test_a_missing_sdk_is_reported_loudly(self, caplog):
        """Believing you have traces and not having them is the worst case."""
        import builtins

        from src.observability import tracing_otel as otel

        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name.startswith("opentelemetry"):
                raise ImportError("no opentelemetry")
            return real_import(name, *args, **kwargs)

        with patch("src.observability.tracing_otel.settings") as s:
            s.otel_enabled = True
            with patch.object(builtins, "__import__", blocked):
                with caplog.at_level(logging.ERROR):
                    assert otel.setup_tracing() is False
        assert "OTEL_ENABLED=true" in caplog.text


class TestObservabilityConfigDefaults:
    def test_tracing_ships_disabled(self):
        """So the SDK stays an optional dependency in practice."""
        import inspect
        import re

        import config

        source = inspect.getsource(config.Settings)
        assert re.search(r'otel_enabled:.*"OTEL_ENABLED",\s*"false"', source)
