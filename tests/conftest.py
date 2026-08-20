"""Shared fixtures for the test suite."""
from __future__ import annotations

import socket
import sys
from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable


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
def _disable_api_rate_limit(request):
    """Disable slowapi rate limiting during tests.

    Endpoint limits (e.g. HEAVY_RATE_LIMIT=5/minute on /upload) would
    otherwise return 429 partway through test modules that call the same
    endpoint repeatedly. Only touches the limiter if api.app is already
    imported, so pure unit tests don't pay the FastAPI import cost.

    Because this is autouse, *no* test exercised the limiter's own code
    path — which is how ``headers_enabled=True`` shipped and made every
    rate-limited endpoint return 500, found by a load test rather than by
    the suite. Mark a test ``@pytest.mark.rate_limited`` to opt back in.
    """
    app_module = sys.modules.get("api.app")
    if app_module is None or request.node.get_closest_marker("rate_limited"):
        yield
        return
    limiter = app_module.limiter
    previous = limiter.enabled
    limiter.enabled = False
    yield
    limiter.enabled = previous


@pytest.fixture(autouse=True)
def _no_outbound_network(request):
    """Block outbound connections in the unit suite.

    Unit tests fake the LLM by patching ``settings`` and expecting the
    client to be inert. That assumption is only as good as whatever the
    client does with a fake key — and it broke silently once model
    construction moved into ``src.llm.providers``, which reads the *real*
    settings: mocked tests began issuing real, billed OpenAI requests and
    the suite slowed from 90 seconds to 57 minutes.

    Rather than rely on every future test remembering to mock at the right
    seam, connections to anything but the loopback interface raise here.
    A test that needs a real backend must be marked ``integration``,
    where the containers live on localhost anyway.
    """
    if request.node.get_closest_marker("integration"):
        yield
        return

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def _local_only(address):
        host = address[0] if isinstance(address, tuple) else address
        return isinstance(host, str) and (
            host in {"localhost", "::1"} or host.startswith("127.")
        )

    def guarded_connect(self, address, *args, **kwargs):
        if not _local_only(address):
            raise RuntimeError(
                f"Blocked outbound connection to {address!r} from a unit test. "
                "Mock the client, or mark the test with @pytest.mark.integration."
            )
        return real_connect(self, address, *args, **kwargs)

    def guarded_connect_ex(self, address, *args, **kwargs):
        if not _local_only(address):
            raise RuntimeError(f"Blocked outbound connection to {address!r} from a unit test.")
        return real_connect_ex(self, address, *args, **kwargs)

    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    try:
        yield
    finally:
        socket.socket.connect = real_connect
        socket.socket.connect_ex = real_connect_ex


class _StubChatModel(Runnable):
    """A chat model that answers without leaving the process.

    Covers the three shapes the graph uses: plain invoke, structured
    output, and tool binding.
    """

    def invoke(self, input, config=None, **kwargs):
        return AIMessage(content="stub answer")

    def with_structured_output(self, schema, **kwargs):
        stub = _StubChatModel()
        # Pydantic models in this codebase are all constructible from
        # defaults or a single obvious field; fall back to model_construct
        # so a stub never has to know each schema.
        def _invoke(inp, config=None, **kw):
            try:
                return schema()
            except Exception:
                return schema.model_construct()
        stub.invoke = _invoke  # type: ignore[assignment]
        return stub

    def bind_tools(self, tools, **kwargs):
        return self


@pytest.fixture
def stub_llm():
    """Replace the model factory for tests that exercise whole pipelines.

    Graph tests stub individual nodes, but the graph also runs intent
    detection, query transformation and the analyzer — which reach for a
    model of their own. Those calls used to go to the real API and be
    swallowed by each node's fallback, so the tests passed while quietly
    spending money and taking twenty seconds each.

    Patching the single factory in ``src.llm_pool`` covers every caller,
    which is exactly what the provider abstraction is for.
    """
    from src.llm_pool import reset_pool

    reset_pool()
    with patch("src.llm_pool.build_chat_model", return_value=_StubChatModel()), \
         patch("src.llm_pool.fallback_spec", return_value=None):
        yield
    reset_pool()


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
