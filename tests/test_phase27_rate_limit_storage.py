"""Phase 27 — P0-3: rate limit counters shared across replicas.

The limiter used slowapi's in-memory storage, so counters were
per-process. With N replicas the effective limit was N x the configured
value, and a restart reset it. That capped the deployment at one API
process, which turns any single crash into a full outage.

These tests pin the three things that make the fix real: the URI is
honoured, the key is the caller rather than the shared egress IP, and a
misconfigured backend is reported loudly instead of silently degrading.
"""
from __future__ import annotations

import importlib
import logging
from unittest.mock import MagicMock, patch

import pytest
from limits import parse
from limits.storage import storage_from_string
from limits.strategies import FixedWindowRateLimiter

# ---------------------------------------------------------------------------
# Storage selection
# ---------------------------------------------------------------------------

class TestStorageSelection:
    def test_blank_setting_means_in_process_memory(self):
        import api.rate_limit as rl

        with patch("api.rate_limit.settings") as s:
            s.rate_limit_storage_uri = ""
            assert rl.storage_uri() == "memory://"
            assert rl.is_distributed() is False

    def test_configured_redis_uri_is_used(self):
        import api.rate_limit as rl

        with patch("api.rate_limit.settings") as s:
            s.rate_limit_storage_uri = "redis://cache:6379/0"
            assert rl.storage_uri() == "redis://cache:6379/0"
            assert rl.is_distributed() is True

    def test_surrounding_whitespace_does_not_disable_sharing(self):
        """A trailing space in .env must not silently fall back to memory."""
        import api.rate_limit as rl

        with patch("api.rate_limit.settings") as s:
            s.rate_limit_storage_uri = "  redis://cache:6379/0\n"
            assert rl.is_distributed() is True

    @pytest.mark.parametrize(
        "uri,shared",
        [
            ("redis://h:6379/0", True),
            ("rediss://h:6379/0", True),
            ("redis+sentinel://h:26379/mymaster", True),
            ("memcached://h:11211", True),
            ("memory://", False),
            ("", False),
        ],
    )
    def test_only_real_backends_count_as_shared(self, uri, shared):
        import api.rate_limit as rl

        with patch("api.rate_limit.settings") as s:
            s.rate_limit_storage_uri = uri
            assert rl.is_distributed() is shared

    def test_limiter_is_built_against_the_configured_backend(self):
        import api.rate_limit as rl

        with patch("api.rate_limit.settings") as s:
            s.rate_limit_storage_uri = ""
            limiter = rl.build_limiter()
        # in_memory_fallback_enabled keeps a Redis outage from 500-ing
        # every request; without it Redis becomes a new SPOF.
        assert limiter._in_memory_fallback_enabled is True


# ---------------------------------------------------------------------------
# The actual defect: per-process counters
# ---------------------------------------------------------------------------

class TestCountersAreShared:
    """Two storage handles stand in for two API replicas."""

    LIMIT = parse("5/minute")

    def _drive(self, uri: str, key: str) -> int:
        replica_a = FixedWindowRateLimiter(storage_from_string(uri))
        replica_b = FixedWindowRateLimiter(storage_from_string(uri))
        allowed = 0
        for i in range(20):
            replica = replica_a if i % 2 == 0 else replica_b
            if replica.hit(self.LIMIT, key):
                allowed += 1
        return allowed

    def test_in_memory_storage_lets_each_replica_serve_a_full_quota(self):
        """This is the bug, asserted so a regression is visible."""
        assert self._drive("memory://", "probe-memory") == 10  # 2 x 5

    @pytest.mark.integration
    def test_redis_storage_enforces_one_quota_across_replicas(self, redis_uri):
        assert self._drive(redis_uri, "probe-redis") == 5


# ---------------------------------------------------------------------------
# Who gets rate limited
# ---------------------------------------------------------------------------

class TestRateLimitKey:
    def _request(self, api_key_id="", host="203.0.113.7"):
        req = MagicMock()
        req.state.api_key_id = api_key_id
        req.client.host = host
        req.headers = {}
        return req

    def test_authenticated_callers_are_keyed_by_api_key(self):
        import api.rate_limit as rl

        assert rl.rate_limit_key(self._request(api_key_id="abc123")) == "key:abc123"

    def test_two_keys_behind_one_ip_do_not_share_a_quota(self):
        """Everyone behind a corporate NAT used to exhaust one bucket."""
        import api.rate_limit as rl

        same_ip = "198.51.100.4"
        alice = rl.rate_limit_key(self._request("alice-hash", same_ip))
        bob = rl.rate_limit_key(self._request("bob-hash", same_ip))
        assert alice != bob

    def test_unauthenticated_falls_back_to_client_address(self):
        import api.rate_limit as rl

        assert rl.rate_limit_key(self._request(host="203.0.113.9")) == "203.0.113.9"

    def test_no_raw_credential_is_used_as_a_key(self):
        """api_key_id is a truncated hash, so Redis never sees the secret."""
        from src.security.auth import api_key_identity

        secret = "super-secret-api-key"
        identity = api_key_identity(secret)
        assert secret not in identity
        assert len(identity) == 16


# ---------------------------------------------------------------------------
# Startup probe
# ---------------------------------------------------------------------------

class TestStartupProbe:
    def test_unreachable_backend_is_reported_as_an_error(self, caplog):
        """Silent fallback would look healthy while limits stopped applying."""
        import api.rate_limit as rl

        limiter = MagicMock()
        limiter.limiter.storage.check.side_effect = ConnectionError("refused")

        with patch("api.rate_limit.settings") as s:
            s.rate_limit_storage_uri = "redis://cache:6379/0"
            with caplog.at_level(logging.ERROR):
                assert rl.verify_storage(limiter) is False
        assert "unreachable" in caplog.text

    def test_failed_health_check_is_reported(self, caplog):
        import api.rate_limit as rl

        limiter = MagicMock()
        limiter.limiter.storage.check.return_value = False

        with patch("api.rate_limit.settings") as s:
            s.rate_limit_storage_uri = "redis://cache:6379/0"
            with caplog.at_level(logging.ERROR):
                assert rl.verify_storage(limiter) is False
        assert "health check" in caplog.text

    def test_healthy_backend_passes(self):
        import api.rate_limit as rl

        limiter = MagicMock()
        limiter.limiter.storage.check.return_value = True

        with patch("api.rate_limit.settings") as s:
            s.rate_limit_storage_uri = "redis://cache:6379/0"
            assert rl.verify_storage(limiter) is True

    def test_memory_storage_warns_when_auth_is_on(self, caplog):
        """Auth on implies a real deployment, where one replica is a risk."""
        import api.rate_limit as rl

        with patch("api.rate_limit.settings") as s:
            s.rate_limit_storage_uri = ""
            s.auth_enabled = True
            with caplog.at_level(logging.WARNING):
                assert rl.verify_storage(MagicMock()) is True
        assert "not shared" in caplog.text

    def test_memory_storage_is_quiet_in_single_tenant_dev(self, caplog):
        import api.rate_limit as rl

        with patch("api.rate_limit.settings") as s:
            s.rate_limit_storage_uri = ""
            s.auth_enabled = False
            with caplog.at_level(logging.WARNING):
                assert rl.verify_storage(MagicMock()) is True
        assert caplog.text == ""

    @pytest.mark.parametrize(
        "uri,expected",
        [
            ("redis://user:hunter2@cache:6379/0", "redis://***@cache:6379/0"),
            ("redis://:hunter2@cache:6379/0", "redis://***@cache:6379/0"),
            ("redis://cache:6379/0", "redis://cache:6379/0"),
        ],
    )
    def test_passwords_are_redacted_before_logging(self, uri, expected):
        import api.rate_limit as rl

        assert rl._redact(uri) == expected


# ---------------------------------------------------------------------------
# The limiter's own code path
# ---------------------------------------------------------------------------

@pytest.mark.rate_limited
class TestLimiterRunsOnRealRequests:
    """With the limiter ENABLED — the path the autouse fixture hides.

    ``headers_enabled=True`` makes slowapi write X-RateLimit-* into an
    injected ``response`` parameter, and raise if the endpoint does not
    declare one. Every rate-limited endpoint returned 500 until three of
    them gained that parameter, and 1,600 passing tests said nothing
    because the limiter was disabled in all of them.
    """

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        s = MagicMock()
        s.validate = MagicMock()
        s.auth_enabled = False
        s.chroma_collection = "c"
        s.log_level = "WARNING"
        s.debug_mode = False
        s.max_upload_size_mb = 10
        s.cors_origins = "http://localhost:8501"
        s.cors_allow_methods = "GET,POST,OPTIONS"
        s.cors_allow_headers = "Content-Type"
        s.async_ingestion = False
        s.guardrails_enabled = False
        s.rate_limit_storage_uri = ""
        s.rate_limit_per_minute = "100/minute"
        s.heavy_rate_limit = "100/minute"
        with patch("api.app.settings", s), patch("src.security.auth.settings", s):
            from api.app import app

            with TestClient(app) as tc:
                yield tc

    def _answer(self):
        from api.models import AskResponse

        return AskResponse(
            answer="ok", question="q", mode="naive", retriever_strategy="dense",
            cost_usd=0.0, latency_ms=1.0, tokens_used=0,
        )

    def test_ask_succeeds_with_the_limiter_active(self, client):
        with patch("api.app._ask_sync", return_value=self._answer()):
            resp = client.post("/ask", json={"question": "hello?"})
        assert resp.status_code == 200, resp.text

    def test_quota_headers_reach_the_client(self, client):
        """The reason headers_enabled is on: clients can back off."""
        with patch("api.app._ask_sync", return_value=self._answer()):
            resp = client.post("/ask", json={"question": "hello?"})
        assert any(h.lower().startswith("x-ratelimit") for h in resp.headers)

    def test_ingest_succeeds_with_the_limiter_active(self, client):
        from api.models import IngestResponse

        with patch(
            "api.app._ingest_sync",
            return_value=IngestResponse(
                documents_loaded=1, chunks_created=1, chunks_added=1, collection_total=1
            ),
        ):
            resp = client.post("/ingest", json={"path": "./data/sample_docs"})
        assert resp.status_code != 500, resp.text

    def test_upload_succeeds_with_the_limiter_active(self, client):
        from api.models import UploadResponse

        with patch(
            "api.app._upload_sync",
            return_value=UploadResponse(
                filename="a.txt", department="general", documents_loaded=1,
                chunks_created=1, chunks_added=1, collection_total=1,
            ),
        ):
            resp = client.post(
                "/upload",
                files={"file": ("a.txt", b"hello", "text/plain")},
                data={"department": "general"},
            )
        assert resp.status_code == 200, resp.text

    def test_every_limited_endpoint_declares_a_response_parameter(self):
        """The structural version of the same check.

        slowapi injects quota headers through this parameter, so an
        endpoint without one fails at request time, not import time.
        """
        import inspect
        import re

        import api.app

        source = inspect.getsource(api.app)
        offenders = []
        for match in re.finditer(
            r"@limiter\.limit\([^)]*\)\s*\n\s*async def (\w+)\(([^)]*)\)",
            source,
        ):
            name, params = match.group(1), match.group(2)
            if "response" not in params:
                offenders.append(name)
        assert not offenders, (
            f"rate-limited endpoints missing a `response` parameter: {offenders} "
            "— these return 500 whenever the limiter is enabled"
        )


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def _settings(self, **over):
        import config

        base = {
            "openai_api_key": "sk-test",
            "rate_limit_storage_uri": "",
        }
        base.update(over)
        return config.Settings(**base)

    def test_a_typo_in_the_scheme_is_rejected_at_startup(self):
        """`redis:/host` would otherwise silently mean 'no sharing'."""
        with pytest.raises(ValueError, match="RATE_LIMIT_STORAGE_URI"):
            self._settings(rate_limit_storage_uri="redis:/cache:6379").validate()

    @pytest.mark.parametrize(
        "uri", ["redis://cache:6379/0", "rediss://cache:6380/0", "memory://", ""]
    )
    def test_valid_uris_are_accepted(self, uri):
        self._settings(rate_limit_storage_uri=uri).validate()

    def test_the_shipped_default_is_in_process(self):
        """Asserted against source, not the developer's .env."""
        import inspect
        import re

        import config

        source = inspect.getsource(config.Settings)
        assert re.search(r'rate_limit_storage_uri:\s*str\s*=\s*os\.getenv\(\s*"RATE_LIMIT_STORAGE_URI",\s*""\s*\)', source)
