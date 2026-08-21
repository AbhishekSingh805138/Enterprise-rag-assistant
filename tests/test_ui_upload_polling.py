"""The upload UI reports when indexing actually finishes.

With ASYNC_INGESTION the API returns 202 and a worker indexes the file in
another process. The sidebar said "accepted for indexing", printed the
status URL and stopped there — so a stalled worker and a slow one looked
identical, and the only way to find out which was to ask a question and
see whether the answer knew about the document.

Polling is time-dependent code, which is exactly the shape that produced
a flaky test elsewhere in this suite, so the clock and the sleep are
injected rather than real. These tests run in microseconds and cannot
race.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import requests

UI_PATH = Path(__file__).resolve().parent.parent / "ui" / "app.py"


class _Resp:
    def __init__(self, payload=None, ok=True):
        self._payload = payload or {}
        self.ok = ok
        self.text = ""

    def json(self):
        return self._payload


@pytest.fixture
def ui(monkeypatch):
    """Load ui/app.py with its import-time /health call stubbed.

    The module is a Streamlit script: importing it runs the whole page
    body, which includes a live request to /health. Left alone that makes
    the test suite depend on a running API.
    """
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(ok=False))
    spec = importlib.util.spec_from_file_location("ui_app_under_test", UI_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Clock:
    """Monotonic clock that only advances when the code sleeps."""

    def __init__(self):
        self.t = 0.0
        self.sleeps: list[float] = []

    def monotonic(self):
        return self.t

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.t += seconds


def _responder(ui, statuses, monkeypatch, errors_before=0):
    """Serve *statuses* in order; the last one repeats forever."""
    calls = {"n": 0}

    def _get(url, **kwargs):
        calls["n"] += 1
        if calls["n"] <= errors_before:
            raise requests.ConnectionError("worker host briefly unreachable")
        i = min(calls["n"] - errors_before - 1, len(statuses) - 1)
        return _Resp({"document_id": "doc_x", "status": statuses[i], "chunks_indexed": 3})

    monkeypatch.setattr(ui.requests, "get", _get)
    return calls


class TestPollDocumentStatus:
    def test_a_processed_document_returns_on_the_first_poll(self, ui, monkeypatch):
        clock = _Clock()
        calls = _responder(ui, ["PROCESSED"], monkeypatch)
        record = ui._poll_document_status(
            "doc_x", sleep=clock.sleep, monotonic=clock.monotonic
        )
        assert record["status"] == "PROCESSED"
        assert calls["n"] == 1

    def test_it_does_not_sleep_once_the_status_is_terminal(self, ui, monkeypatch):
        """Sleeping after the answer is known just delays the UI."""
        clock = _Clock()
        _responder(ui, ["PROCESSED"], monkeypatch)
        ui._poll_document_status("doc_x", sleep=clock.sleep, monotonic=clock.monotonic)
        assert clock.sleeps == []

    def test_it_keeps_polling_while_the_document_is_pending(self, ui, monkeypatch):
        clock = _Clock()
        calls = _responder(ui, ["PENDING", "PENDING", "PROCESSED"], monkeypatch)
        record = ui._poll_document_status(
            "doc_x", sleep=clock.sleep, monotonic=clock.monotonic
        )
        assert record["status"] == "PROCESSED"
        assert calls["n"] == 3

    def test_a_failed_document_is_terminal(self, ui, monkeypatch):
        """A failure must surface, not spin until the deadline."""
        clock = _Clock()
        calls = _responder(ui, ["FAILED"], monkeypatch)
        record = ui._poll_document_status(
            "doc_x", sleep=clock.sleep, monotonic=clock.monotonic
        )
        assert record["status"] == "FAILED"
        assert calls["n"] == 1

    def test_a_dead_lettered_document_is_terminal(self, ui, monkeypatch):
        clock = _Clock()
        _responder(ui, ["DEAD_LETTER"], monkeypatch)
        record = ui._poll_document_status(
            "doc_x", sleep=clock.sleep, monotonic=clock.monotonic
        )
        assert record["status"] == "DEAD_LETTER"

    def test_it_gives_up_at_the_deadline(self, ui, monkeypatch):
        """A worker that never runs must not hang the sidebar forever."""
        clock = _Clock()
        _responder(ui, ["PENDING"], monkeypatch)
        record = ui._poll_document_status(
            "doc_x", timeout_s=10.0, interval_s=2.0,
            sleep=clock.sleep, monotonic=clock.monotonic,
        )
        assert record["status"] == "PENDING"  # last known, not a terminal one
        assert clock.t <= 10.0 + 2.0

    def test_a_transient_error_does_not_abort_the_poll(self, ui, monkeypatch):
        """The API restarting mid-index should not read as a failure."""
        clock = _Clock()
        _responder(ui, ["PROCESSED"], monkeypatch, errors_before=2)
        record = ui._poll_document_status(
            "doc_x", timeout_s=30.0, interval_s=1.0,
            sleep=clock.sleep, monotonic=clock.monotonic,
        )
        assert record["status"] == "PROCESSED"

    def test_an_unreachable_api_returns_no_record(self, ui, monkeypatch):
        """None means 'never observed', which the caller reports honestly."""
        clock = _Clock()

        def _always_fail(url, **kwargs):
            raise requests.ConnectionError("down")

        monkeypatch.setattr(ui.requests, "get", _always_fail)
        record = ui._poll_document_status(
            "doc_x", timeout_s=5.0, interval_s=1.0,
            sleep=clock.sleep, monotonic=clock.monotonic,
        )
        assert record is None
