"""Shared fixtures for the test suite."""
from __future__ import annotations

import sys

import pytest
from langchain_core.documents import Document


@pytest.fixture(autouse=True)
def _reset_circuit_breakers():
    """Reset global circuit breakers between tests.

    Breakers are process-global singletons (src.resilience.circuit_breaker).
    A test that trips one (e.g. the "retrieval" or "llm" breaker) would
    otherwise leave it OPEN, causing unrelated later tests to short-circuit
    and fail depending on collection order. Clear before and after each test
    so runs are order-independent.
    """
    from src.resilience.circuit_breaker import reset_all_breakers

    reset_all_breakers()
    yield
    reset_all_breakers()


@pytest.fixture(autouse=True)
def _disable_api_rate_limit():
    """Disable slowapi rate limiting during tests.

    Endpoint limits (e.g. HEAVY_RATE_LIMIT=5/minute on /upload) would
    otherwise return 429 partway through test modules that call the same
    endpoint repeatedly. Only touches the limiter if api.app is already
    imported, so pure unit tests don't pay the FastAPI import cost.
    """
    app_module = sys.modules.get("api.app")
    if app_module is None:
        yield
        return
    limiter = app_module.limiter
    previous = limiter.enabled
    limiter.enabled = False
    yield
    limiter.enabled = previous


@pytest.fixture
def redis_uri() -> str:
    """URI of the Redis in docker-compose.test.yml, skipping if it is down.

    Used by tests marked ``integration``; those are excluded by default so
    the everyday suite needs no containers.
    """
    import os

    uri = os.getenv("TEST_REDIS_URL", "redis://localhost:6380/0")
    try:
        import redis

        redis.Redis.from_url(uri, socket_connect_timeout=2).ping()
    except Exception as e:
        pytest.skip(f"Redis not reachable at {uri}: {e}")
    return uri


@pytest.fixture
def sample_documents() -> list[Document]:
    """A small set of Document objects for unit tests (no API calls needed)."""
    return [
        Document(
            page_content="Employees may work remotely up to 3 days per week with manager approval.",
            metadata={
                "source": "/data/hr/handbook.md",
                "filename": "handbook.md",
                "doc_type": "md",
                "department": "hr",
                "access_level": "internal",
                "start_index": 0,
            },
        ),
        Document(
            page_content="Standard payment terms are Net 30 from the date of invoice.",
            metadata={
                "source": "/data/legal/vendor_contract_terms.md",
                "filename": "vendor_contract_terms.md",
                "doc_type": "md",
                "department": "legal",
                "access_level": "confidential",
                "start_index": 0,
            },
        ),
        Document(
            page_content="All API endpoints must require authentication using OAuth 2.0 with JWT.",
            metadata={
                "source": "/data/engineering/api_guidelines.md",
                "filename": "api_guidelines.md",
                "doc_type": "md",
                "department": "engineering",
                "access_level": "internal",
                "start_index": 0,
            },
        ),
    ]


@pytest.fixture
def sample_docs_path(tmp_path) -> str:
    """Create a temporary directory with sample documents for loader tests."""
    # Create department subdirs
    hr = tmp_path / "hr"
    hr.mkdir()
    (hr / "policy.md").write_text(
        "# Leave Policy\nEmployees get 20 days PTO per year.\n",
        encoding="utf-8",
    )
    (hr / "handbook.txt").write_text(
        "Remote work is allowed 3 days per week.\n",
        encoding="utf-8",
    )

    legal = tmp_path / "legal"
    legal.mkdir()
    (legal / "terms.md").write_text(
        "# Vendor Terms\nPayment terms are Net 30.\nSLA uptime is 99.9%.\n",
        encoding="utf-8",
    )

    # A file at root level (no department subfolder)
    (tmp_path / "readme.md").write_text(
        "# Company Overview\nAcme Corp was founded in 2010.\n",
        encoding="utf-8",
    )

    return str(tmp_path)
