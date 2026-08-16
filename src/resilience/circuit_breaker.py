"""Circuit breaker pattern for external service calls.

Protects the system from cascading failures when external services
(LLM, ChromaDB, Tavily) are down. The breaker has three states:

  CLOSED   — normal operation; failures are counted.
  OPEN     — service is assumed down; calls fail fast without trying.
  HALF_OPEN — after a cooldown, one probe request is allowed through.

Transition rules:
  CLOSED  -> OPEN      when failures >= threshold within the window.
  OPEN    -> HALF_OPEN when timeout has elapsed.
  HALF_OPEN -> CLOSED  if the probe succeeds.
  HALF_OPEN -> OPEN    if the probe fails (resets the timeout).
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpen(Exception):
    """Raised when a call is attempted on an open circuit breaker."""

    def __init__(self, name: str, remaining_seconds: float) -> None:
        self.name = name
        self.remaining_seconds = remaining_seconds
        super().__init__(
            f"Circuit breaker '{name}' is OPEN — retry in {remaining_seconds:.0f}s"
        )


class CircuitBreaker:
    """Thread-safe circuit breaker for external service calls.

    Args:
        name: Human-readable name (e.g. "llm", "chromadb", "tavily").
        failure_threshold: Number of failures before opening the circuit.
        timeout: Seconds to wait in OPEN state before probing (HALF_OPEN).
        success_threshold: Consecutive successes in HALF_OPEN to fully close.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        success_threshold: int = 1,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.success_threshold = success_threshold

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.monotonic() - self._last_failure_time >= self.timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    logger.info(
                        "Circuit '%s' → HALF_OPEN (timeout elapsed)", self.name
                    )
            return self._state

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info("Circuit '%s' → CLOSED (probe succeeded)", self.name)
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success in closed state
                self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self._last_failure_time = time.monotonic()
            if self._state == CircuitState.HALF_OPEN:
                # Probe failed — go back to OPEN
                self._state = CircuitState.OPEN
                self._success_count = 0
                logger.warning(
                    "Circuit '%s' → OPEN (probe failed)", self.name
                )
            elif self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.warning(
                        "Circuit '%s' → OPEN (%d failures)",
                        self.name,
                        self._failure_count,
                    )

    def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute *fn* through the circuit breaker.

        Raises CircuitBreakerOpen if the circuit is open.
        """
        current_state = self.state  # triggers OPEN -> HALF_OPEN transition

        if current_state == CircuitState.OPEN:
            remaining = self.timeout - (
                time.monotonic() - self._last_failure_time
            )
            raise CircuitBreakerOpen(self.name, max(remaining, 0))

        try:
            result = fn(*args, **kwargs)
            self.record_success()
            return result
        except CircuitBreakerOpen:
            raise
        except Exception:
            self.record_failure()
            raise

    def reset(self) -> None:
        """Manually reset the breaker to CLOSED (for testing/admin)."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = 0.0
            logger.info("Circuit '%s' manually reset to CLOSED", self.name)

    def stats(self) -> dict:
        """Return current breaker stats."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "failure_threshold": self.failure_threshold,
                "timeout": self.timeout,
            }


# ---------------------------------------------------------------------------
# Pre-configured breakers for known external services
# ---------------------------------------------------------------------------

_breakers: dict[str, CircuitBreaker] = {}
_breakers_lock = threading.Lock()


def get_breaker(
    name: str,
    failure_threshold: int = 5,
    timeout: float = 60.0,
) -> CircuitBreaker:
    """Return a named circuit breaker (singleton per name). Thread-safe."""
    with _breakers_lock:
        if name not in _breakers:
            _breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                timeout=timeout,
            )
        return _breakers[name]


def reset_all_breakers() -> None:
    """Reset all breakers (for testing)."""
    with _breakers_lock:
        for b in _breakers.values():
            b.reset()
        _breakers.clear()


def all_breaker_stats() -> list[dict]:
    """Return stats for all active breakers."""
    with _breakers_lock:
        return [b.stats() for b in _breakers.values()]
