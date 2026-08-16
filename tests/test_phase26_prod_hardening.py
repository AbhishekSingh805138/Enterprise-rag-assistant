"""Phase 26 — production hardening (P0 items from the readiness review).

Two changes with security consequences:

  * ``GET /health?deep=true`` used to be unauthenticated while reporting
    the vector store endpoint, queue depths, registry counts and process
    memory — internal topology readable by anything that could reach the
    port. The shallow probe stays open so load balancers still work.
  * Secrets can now be read from a mounted file (``<NAME>_FILE``) instead
    of the process environment, where ``docker inspect`` and crash dumps
    can reach them.
"""
from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

VALID_KEY = "test-key-abcdefghijklmnop"


# ---------------------------------------------------------------------------
# Deep health authentication
# ---------------------------------------------------------------------------

def make_client(*, auth_enabled: bool):
    """TestClient with auth on or off, and the deep check fully stubbed."""
    mock = MagicMock()
    mock.validate = MagicMock()
    mock.chroma_collection = "test_collection"
    mock.log_level = "WARNING"
    mock.debug_mode = False
    mock.max_upload_size_mb = 10
    mock.cors_origins = "http://localhost:8501"
    mock.cors_allow_methods = "GET,POST,OPTIONS"
    mock.cors_allow_headers = "Content-Type,Authorization,X-Request-ID"
    mock.ingest_root = "./data"
    mock.async_ingestion = False
    mock.auth_enabled = auth_enabled
    mock.api_keys = VALID_KEY
    return mock


@pytest.fixture
def auth_client():
    """API with AUTH_ENABLED=true."""
    settings_mock = make_client(auth_enabled=True)
    with (
        patch("api.app.settings", settings_mock),
        patch("src.security.auth.settings", settings_mock),
    ):
        from api.app import app

        with TestClient(app) as tc:
            yield tc


@pytest.fixture
def open_client():
    """API with AUTH_ENABLED=false (single-tenant dev mode)."""
    settings_mock = make_client(auth_enabled=False)
    with (
        patch("api.app.settings", settings_mock),
        patch("src.security.auth.settings", settings_mock),
    ):
        from api.app import app

        with TestClient(app) as tc:
            yield tc


def stub_deep_check():
    from src.observability.health_checker import DeepHealthResult, HealthCheck

    return patch(
        "src.observability.health_checker.deep_health_check",
        return_value=DeepHealthResult(
            status="ok",
            checks=[
                HealthCheck(
                    name="chromadb", status="ok", latency_ms=1.0,
                    detail="114 documents in enterprise_docs (server: http://10.0.0.5:8000)",
                ),
            ],
        ),
    )


class TestShallowHealthStaysPublic:
    """Load balancers and orchestrators probe this without credentials."""

    def test_shallow_health_needs_no_key(self, auth_client):
        with patch(
            "src.vectorstore.chroma_store.collection_stats",
            return_value={"collection": "test_collection", "document_count": 5},
        ):
            resp = auth_client.get("/health")
        assert resp.status_code == 200

    def test_shallow_health_exposes_no_topology(self, auth_client):
        with patch(
            "src.vectorstore.chroma_store.collection_stats",
            return_value={"collection": "test_collection", "document_count": 5},
        ):
            body = auth_client.get("/health").json()
        assert "checks" not in body
        assert "endpoint" not in str(body)

    def test_shallow_health_still_reports_when_the_store_is_down(self, auth_client):
        with patch(
            "src.vectorstore.chroma_store.collection_stats",
            side_effect=RuntimeError("chroma unreachable"),
        ):
            resp = auth_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "degraded"


class TestDeepHealthRequiresAuth:
    def test_deep_health_without_a_key_is_rejected(self, auth_client):
        with stub_deep_check():
            resp = auth_client.get("/health?deep=true")
        assert resp.status_code == 401

    def test_rejection_leaks_no_internal_detail(self, auth_client):
        """The 401 body must not carry the topology it is protecting."""
        with stub_deep_check():
            body = auth_client.get("/health?deep=true").text
        assert "10.0.0.5" not in body
        assert "enterprise_docs" not in body

    def test_deep_health_with_a_valid_key_succeeds(self, auth_client):
        with stub_deep_check():
            resp = auth_client.get(
                "/health?deep=true", headers={"Authorization": f"Bearer {VALID_KEY}"}
            )
        assert resp.status_code == 200
        assert resp.json()["checks"][0]["name"] == "chromadb"

    def test_deep_health_with_a_bad_key_is_rejected(self, auth_client):
        with stub_deep_check():
            resp = auth_client.get(
                "/health?deep=true", headers={"Authorization": "Bearer wrong-key"}
            )
        assert resp.status_code == 401

    def test_malformed_authorization_header_is_rejected(self, auth_client):
        with stub_deep_check():
            resp = auth_client.get(
                "/health?deep=true", headers={"Authorization": VALID_KEY}  # no "Bearer"
            )
        assert resp.status_code == 401

    def test_deep_health_stays_open_when_auth_is_disabled(self, open_client):
        """Dev mode must not need a key to inspect subsystems."""
        with stub_deep_check():
            resp = open_client.get("/health?deep=true")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# File-based secrets
# ---------------------------------------------------------------------------

class TestSecretResolution:
    def _secret(self):
        import config

        return config._secret

    def test_falls_back_to_the_environment_variable(self, monkeypatch):
        monkeypatch.delenv("DEMO_SECRET_FILE", raising=False)
        monkeypatch.setenv("DEMO_SECRET", "from-env")
        assert self._secret()("DEMO_SECRET") == "from-env"

    def test_default_when_neither_is_set(self, monkeypatch):
        monkeypatch.delenv("DEMO_SECRET_FILE", raising=False)
        monkeypatch.delenv("DEMO_SECRET", raising=False)
        assert self._secret()("DEMO_SECRET", "fallback") == "fallback"

    def test_file_wins_over_the_environment_variable(self, tmp_path, monkeypatch):
        """A mounted secret must beat a stale value left in the environment."""
        path = tmp_path / "secret"
        path.write_text("from-file", encoding="utf-8")
        monkeypatch.setenv("DEMO_SECRET", "from-env")
        monkeypatch.setenv("DEMO_SECRET_FILE", str(path))
        assert self._secret()("DEMO_SECRET") == "from-file"

    def test_surrounding_whitespace_is_stripped(self, tmp_path, monkeypatch):
        """Secret managers commonly append a trailing newline."""
        path = tmp_path / "secret"
        path.write_text("  sk-with-newline\n", encoding="utf-8")
        monkeypatch.setenv("DEMO_SECRET_FILE", str(path))
        assert self._secret()("DEMO_SECRET") == "sk-with-newline"

    def test_unreadable_file_fails_loudly(self, tmp_path, monkeypatch):
        """Silently falling back would start the app with the wrong credential."""
        monkeypatch.setenv("DEMO_SECRET", "from-env")
        monkeypatch.setenv("DEMO_SECRET_FILE", str(tmp_path / "does-not-exist"))
        with pytest.raises(RuntimeError, match="could not be read"):
            self._secret()("DEMO_SECRET")

    def test_empty_file_fails_loudly(self, tmp_path, monkeypatch):
        path = tmp_path / "secret"
        path.write_text("   \n", encoding="utf-8")
        monkeypatch.setenv("DEMO_SECRET_FILE", str(path))
        with pytest.raises(RuntimeError, match="is empty"):
            self._secret()("DEMO_SECRET")

    def test_blank_file_variable_is_ignored(self, monkeypatch):
        """An unset-but-present FILE var must not shadow the plain one."""
        monkeypatch.setenv("DEMO_SECRET_FILE", "   ")
        monkeypatch.setenv("DEMO_SECRET", "from-env")
        assert self._secret()("DEMO_SECRET") == "from-env"

    @pytest.mark.parametrize(
        "field,env_var",
        [
            ("openai_api_key", "OPENAI_API_KEY"),
            ("api_keys", "API_KEYS"),
            ("tavily_api_key", "TAVILY_API_KEY"),
            ("langsmith_api_key", "LANGSMITH_API_KEY"),
            ("chroma_auth_token", "CHROMA_AUTH_TOKEN"),
        ],
    )
    def test_every_credential_supports_the_file_convention(self, field, env_var):
        """Any secret left on the env-only path would undermine the control."""
        import inspect
        import re

        import config

        source = inspect.getsource(config.Settings)
        assert re.search(rf'{field}:\s*str\s*=\s*_secret\("{env_var}"', source), (
            f"{field} does not use _secret() and cannot be supplied as a file"
        )

    def test_settings_reads_a_mounted_key_end_to_end(self, tmp_path, monkeypatch):
        import config

        path = tmp_path / "openai_key"
        path.write_text("sk-mounted-from-a-secret-volume", encoding="utf-8")
        monkeypatch.setenv("OPENAI_API_KEY_FILE", str(path))
        reloaded = importlib.reload(config)
        try:
            assert reloaded.Settings().openai_api_key == "sk-mounted-from-a-secret-volume"
        finally:
            monkeypatch.delenv("OPENAI_API_KEY_FILE", raising=False)
            importlib.reload(config)
