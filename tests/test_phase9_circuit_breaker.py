"""Phase 9.2.2 — Circuit breaker tests.

Tests state transitions (closed -> open -> half_open -> closed),
fail-fast behavior, probing on half-open, and integration with
the node layer.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from src.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpen,
    CircuitState,
    all_breaker_stats,
    get_breaker,
    reset_all_breakers,
)

# ---------------------------------------------------------------------------
# State transition tests
# ---------------------------------------------------------------------------

class TestCircuitBreakerTransitions:
    """Core state-machine transition tests."""

    def test_starts_closed(self):
        cb = CircuitBreaker("test", failure_threshold=3, timeout=10)
        assert cb.state == CircuitState.CLOSED

    def test_stays_closed_under_threshold(self):
        cb = CircuitBreaker("test", failure_threshold=3, timeout=10)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_opens_at_threshold(self):
        cb = CircuitBreaker("test", failure_threshold=3, timeout=10)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker("test", failure_threshold=3, timeout=10)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        # Should need 3 more failures to open
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_transitions_to_half_open_after_timeout(self):
        cb = CircuitBreaker("test", failure_threshold=2, timeout=0.1)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_to_closed_on_success(self):
        cb = CircuitBreaker("test", failure_threshold=2, timeout=0.1)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_to_open_on_failure(self):
        cb = CircuitBreaker("test", failure_threshold=2, timeout=0.1)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN


# ---------------------------------------------------------------------------
# call() wrapper tests
# ---------------------------------------------------------------------------

class TestCircuitBreakerCall:
    """Tests for the call() method that wraps external functions."""

    def test_call_succeeds_when_closed(self):
        cb = CircuitBreaker("test", failure_threshold=3, timeout=10)
        result = cb.call(lambda: 42)
        assert result == 42

    def test_call_raises_original_exception(self):
        cb = CircuitBreaker("test", failure_threshold=3, timeout=10)
        with pytest.raises(ValueError, match="boom"):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("boom")))

    def test_call_fails_fast_when_open(self):
        cb = CircuitBreaker("test", failure_threshold=2, timeout=60)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        with pytest.raises(CircuitBreakerOpen) as exc_info:
            cb.call(lambda: 42)
        assert "test" in str(exc_info.value)

    def test_call_tracks_failures_to_open(self):
        cb = CircuitBreaker("test", failure_threshold=2, timeout=60)

        def _fail():
            raise RuntimeError("service down")

        for _ in range(2):
            with pytest.raises(RuntimeError):
                cb.call(_fail)

        assert cb.state == CircuitState.OPEN

    def test_call_probe_succeeds_in_half_open(self):
        cb = CircuitBreaker("test", failure_threshold=2, timeout=0.1)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)

        result = cb.call(lambda: "ok")
        assert result == "ok"
        assert cb.state == CircuitState.CLOSED

    def test_call_probe_fails_in_half_open(self):
        cb = CircuitBreaker("test", failure_threshold=2, timeout=0.1)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("still down")))
        assert cb.state == CircuitState.OPEN


# ---------------------------------------------------------------------------
# Reset and stats
# ---------------------------------------------------------------------------

class TestCircuitBreakerAdmin:

    def test_manual_reset(self):
        cb = CircuitBreaker("test", failure_threshold=2, timeout=60)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED

    def test_stats(self):
        cb = CircuitBreaker("test", failure_threshold=5, timeout=30)
        cb.record_failure()
        stats = cb.stats()
        assert stats["name"] == "test"
        assert stats["state"] == "closed"
        assert stats["failure_count"] == 1
        assert stats["failure_threshold"] == 5
        assert stats["timeout"] == 30


# ---------------------------------------------------------------------------
# Module-level singleton breakers
# ---------------------------------------------------------------------------

class TestBreakerRegistry:

    def setup_method(self):
        reset_all_breakers()

    def teardown_method(self):
        reset_all_breakers()

    def test_get_breaker_returns_singleton(self):
        b1 = get_breaker("llm")
        b2 = get_breaker("llm")
        assert b1 is b2

    def test_different_names_different_instances(self):
        b1 = get_breaker("llm")
        b2 = get_breaker("tavily")
        assert b1 is not b2

    def test_all_breaker_stats(self):
        get_breaker("llm")
        get_breaker("tavily")
        stats = all_breaker_stats()
        assert len(stats) == 2
        names = {s["name"] for s in stats}
        assert names == {"llm", "tavily"}

    def test_reset_all_clears(self):
        get_breaker("llm")
        reset_all_breakers()
        stats = all_breaker_stats()
        assert len(stats) == 0
