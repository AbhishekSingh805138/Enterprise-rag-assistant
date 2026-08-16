"""Phase 27 — P1-1: retrieval is scoped to the caller's departments.

The department filter arrived in the request body and nothing tied it to
the caller, so any authenticated key could read ``legal`` or ``security``
documents by asking for them. The subtler half of the same defect: a
request with *no* filter searched every department at once, so
confidential material could reach an answer without anyone requesting it.

The tests below are organised around the two ways in and the two ways
round: reading another department, reading everything by omitting the
filter, writing into another department, and losing sparse ranking
because a set-valued filter no longer compares equal.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from src.security.access_control import (
    CONFIDENTIAL_DEPARTMENTS,
    VALID_DEPARTMENTS,
    DepartmentForbidden,
    KeyScopes,
    department_filter,
    enforce_scope,
    matches_filter,
    normalise_requested,
    parse_api_keys,
)


# ---------------------------------------------------------------------------
# Parsing API_KEYS
# ---------------------------------------------------------------------------

class TestKeyParsing:
    def test_plain_keys_stay_unscoped(self):
        """Existing deployments must keep working untouched."""
        scopes = parse_api_keys("alpha,beta")
        assert scopes.keys == frozenset({"alpha", "beta"})
        assert scopes.scope_for("alpha") is None
        assert scopes.any_scoped is False

    def test_scoped_key_is_limited_to_its_departments(self):
        scopes = parse_api_keys("alpha:hr|general")
        assert scopes.scope_for("alpha") == frozenset({"hr", "general"})
        assert scopes.any_scoped is True

    def test_wildcard_is_explicit_unrestricted(self):
        assert parse_api_keys("alpha:*").scope_for("alpha") is None

    def test_mixed_scoped_and_unscoped(self):
        scopes = parse_api_keys("admin,hruser:hr,legaluser:legal|general")
        assert scopes.scope_for("admin") is None
        assert scopes.scope_for("hruser") == frozenset({"hr"})
        assert scopes.scope_for("legaluser") == frozenset({"legal", "general"})

    def test_whitespace_and_empty_entries_are_tolerated(self):
        scopes = parse_api_keys(" alpha:hr , , beta ")
        assert scopes.keys == frozenset({"alpha", "beta"})
        assert scopes.scope_for("alpha") == frozenset({"hr"})

    def test_case_is_normalised_on_departments(self):
        assert parse_api_keys("alpha:HR|Legal").scope_for("alpha") == frozenset({"hr", "legal"})

    def test_a_key_containing_a_colon_is_not_truncated(self):
        """Truncation would leave a shorter secret that still authenticates."""
        scopes = parse_api_keys("prefix:opaque-token-value")
        assert scopes.keys == frozenset({"prefix:opaque-token-value"})
        assert scopes.scope_for("prefix:opaque-token-value") is None

    def test_unknown_department_suffix_is_treated_as_part_of_the_key(self):
        """`key:notadept` must not silently become an unscoped key."""
        scopes = parse_api_keys("alpha:marketing")
        assert scopes.keys == frozenset({"alpha:marketing"})

    def test_partially_valid_spec_is_not_accepted(self):
        """`hr|marketing` half-matching must not grant hr."""
        scopes = parse_api_keys("alpha:hr|marketing")
        assert scopes.keys == frozenset({"alpha:hr|marketing"})

    def test_empty_configuration_yields_no_keys(self):
        assert parse_api_keys("").keys == frozenset()


# ---------------------------------------------------------------------------
# The core defect
# ---------------------------------------------------------------------------

class TestScopeEnforcement:
    HR = frozenset({"hr", "general"})

    def test_unrestricted_caller_is_unchanged(self):
        """Backward compatibility: no scope means the old behaviour."""
        assert enforce_scope({"department": "legal"}, None) == {"department": "legal"}
        assert enforce_scope(None, None) is None

    def test_requesting_a_forbidden_department_is_rejected(self):
        with pytest.raises(DepartmentForbidden) as exc:
            enforce_scope({"department": "legal"}, self.HR)
        assert exc.value.requested == {"legal"}

    def test_permitted_department_passes_through(self):
        assert enforce_scope({"department": "hr"}, self.HR) == {"department": "hr"}

    def test_absent_filter_is_narrowed_rather_than_left_open(self):
        """The quiet half of the bug: no filter meant the whole corpus."""
        enforced = enforce_scope(None, self.HR)
        assert enforced == {"department": {"$in": ["general", "hr"]}}

    def test_empty_filter_is_narrowed_too(self):
        assert enforce_scope({}, self.HR) == {"department": {"$in": ["general", "hr"]}}

    def test_a_non_department_filter_keeps_its_other_fields(self):
        enforced = enforce_scope({"doc_type": "pdf"}, self.HR)
        assert enforced["doc_type"] == "pdf"
        assert enforced["department"] == {"$in": ["general", "hr"]}

    def test_single_department_scope_uses_a_scalar_filter(self):
        assert enforce_scope(None, frozenset({"hr"})) == {"department": "hr"}

    def test_an_in_clause_cannot_smuggle_a_forbidden_department(self):
        """Chroma operator syntax is a second way to name a department."""
        with pytest.raises(DepartmentForbidden):
            enforce_scope({"department": {"$in": ["hr", "legal"]}}, self.HR)

    def test_a_permitted_in_clause_is_allowed(self):
        f = {"department": {"$in": ["hr", "general"]}}
        assert enforce_scope(f, self.HR) == f

    def test_a_list_valued_department_is_checked(self):
        with pytest.raises(DepartmentForbidden):
            enforce_scope({"department": ["hr", "security"]}, self.HR)

    def test_case_variation_does_not_bypass_the_check(self):
        with pytest.raises(DepartmentForbidden):
            enforce_scope({"department": "LEGAL"}, self.HR)

    def test_rejection_is_preferred_over_silently_empty_results(self):
        """An empty answer reads as 'no such policy', which is its own lie."""
        with pytest.raises(DepartmentForbidden):
            enforce_scope({"department": "security"}, self.HR)

    @pytest.mark.parametrize("confidential", sorted(CONFIDENTIAL_DEPARTMENTS))
    def test_confidential_departments_are_not_reachable_by_default_scope(self, confidential):
        allowed = frozenset({"general"})
        with pytest.raises(DepartmentForbidden):
            enforce_scope({"department": confidential}, allowed)
        assert confidential not in enforce_scope(None, allowed)["department"]


class TestNormaliseRequested:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (None, set()),
            ({}, set()),
            ({"doc_type": "pdf"}, set()),
            ({"department": "hr"}, {"hr"}),
            ({"department": "HR"}, {"hr"}),
            ({"department": ["hr", "legal"]}, {"hr", "legal"}),
            ({"department": {"$in": ["hr", "legal"]}}, {"hr", "legal"}),
            ({"department": {"$eq": "finance"}}, {"finance"}),
        ],
    )
    def test_every_filter_shape_is_understood(self, value, expected):
        assert normalise_requested(value) == expected


class TestDepartmentFilter:
    def test_one_department_is_a_scalar(self):
        assert department_filter(frozenset({"hr"})) == {"department": "hr"}

    def test_many_departments_use_in_and_are_sorted(self):
        assert department_filter(frozenset({"hr", "general"})) == {
            "department": {"$in": ["general", "hr"]}
        }


# ---------------------------------------------------------------------------
# The way scoping could have quietly degraded ranking
# ---------------------------------------------------------------------------

class TestFilterMatching:
    """BM25 filters in-process by comparing metadata directly."""

    def test_scalar_equality_still_works(self):
        assert matches_filter({"department": "hr"}, {"department": "hr"})
        assert not matches_filter({"department": "legal"}, {"department": "hr"})

    def test_in_operator_matches(self):
        f = {"department": {"$in": ["general", "hr"]}}
        assert matches_filter({"department": "hr"}, f)
        assert not matches_filter({"department": "legal"}, f)

    def test_scoped_caller_keeps_sparse_ranking(self):
        """Plain `==` against {"$in": [...]} matches nothing, every time.

        Dense retrieval would keep working, so the symptom is quietly
        worse answers for scoped users only — not an error anyone sees.
        """
        corpus = [
            {"department": "hr", "text": "leave policy"},
            {"department": "general", "text": "office hours"},
            {"department": "legal", "text": "severance terms"},
        ]
        scoped = {"department": {"$in": ["general", "hr"]}}
        kept = [m for m in corpus if matches_filter(m, scoped)]
        assert len(kept) == 2
        assert all(m["department"] != "legal" for m in kept)

    def test_no_filter_matches_everything(self):
        assert matches_filter({"department": "legal"}, None)
        assert matches_filter({"department": "legal"}, {})

    def test_missing_metadata_field_does_not_match(self):
        assert not matches_filter({}, {"department": "hr"})

    @pytest.mark.parametrize(
        "op,operand,value,expected",
        [
            ("$eq", "hr", "hr", True),
            ("$eq", "hr", "legal", False),
            ("$ne", "legal", "hr", True),
            ("$ne", "legal", "legal", False),
            ("$nin", ["legal", "security"], "hr", True),
            ("$nin", ["legal", "security"], "legal", False),
        ],
    )
    def test_operators(self, op, operand, value, expected):
        assert matches_filter({"department": value}, {"department": {op: operand}}) is expected

    def test_an_unknown_operator_excludes_rather_than_includes(self, caplog):
        """Failing open here would leak documents the filter meant to hide."""
        with caplog.at_level(logging.WARNING):
            assert matches_filter({"department": "legal"}, {"department": {"$regex": ".*"}}) is False
        assert "Unsupported filter operator" in caplog.text

    def test_all_fields_must_match(self):
        meta = {"department": "hr", "access_level": "internal"}
        assert matches_filter(meta, {"department": "hr", "access_level": "internal"})
        assert not matches_filter(meta, {"department": "hr", "access_level": "confidential"})


# ---------------------------------------------------------------------------
# Resolution from the request
# ---------------------------------------------------------------------------

class TestPermittedDepartments:
    def _request(self, departments=None):
        req = MagicMock()
        req.state.departments = departments
        return req

    def test_auth_disabled_is_unrestricted(self):
        """Single-tenant dev mode has no principal to scope to."""
        from src.security.auth import permitted_departments

        with patch("src.security.auth.settings") as s:
            s.auth_enabled = False
            assert permitted_departments(self._request(frozenset({"hr"}))) is None

    def test_scope_is_read_from_request_state(self):
        from src.security.auth import permitted_departments

        with patch("src.security.auth.settings") as s:
            s.auth_enabled = True
            assert permitted_departments(self._request(frozenset({"hr"}))) == frozenset({"hr"})

    def test_missing_state_defaults_to_unrestricted(self):
        from src.security.auth import permitted_departments

        req = MagicMock(spec=["state"])
        req.state = MagicMock(spec=[])
        with patch("src.security.auth.settings") as s:
            s.auth_enabled = True
            assert permitted_departments(req) is None


class TestAuthAttachesScope:
    """verify_api_key must publish the matched key's scope."""

    @pytest.mark.asyncio
    async def test_scope_is_attached_to_the_matched_key(self):
        from src.security.auth import verify_api_key

        req = MagicMock()
        req.headers = {"Authorization": "Bearer hruser"}
        req.state = MagicMock()

        with patch("src.security.auth.settings") as s:
            s.auth_enabled = True
            s.api_keys = "admin,hruser:hr|general"
            await verify_api_key(req)

        assert req.state.departments == frozenset({"hr", "general"})

    @pytest.mark.asyncio
    async def test_unscoped_key_gets_none(self):
        from src.security.auth import verify_api_key

        req = MagicMock()
        req.headers = {"Authorization": "Bearer admin"}
        req.state = MagicMock()

        with patch("src.security.auth.settings") as s:
            s.auth_enabled = True
            s.api_keys = "admin,hruser:hr"
            await verify_api_key(req)

        assert req.state.departments is None

    @pytest.mark.asyncio
    async def test_the_scope_suffix_is_not_part_of_the_secret(self):
        from fastapi import HTTPException

        from src.security.auth import verify_api_key

        req = MagicMock()
        req.headers = {"Authorization": "Bearer hruser:hr"}  # suffix included
        req.state = MagicMock()

        with patch("src.security.auth.settings") as s:
            s.auth_enabled = True
            s.api_keys = "hruser:hr"
            with pytest.raises(HTTPException) as exc:
                await verify_api_key(req)
        assert exc.value.status_code == 401


# ---------------------------------------------------------------------------
# Taxonomy consistency
# ---------------------------------------------------------------------------

class TestTaxonomy:
    def test_api_and_access_control_share_one_department_list(self):
        """Divergence means uploads that can never be read back."""
        from api.models import VALID_DEPARTMENTS as API_DEPARTMENTS

        assert API_DEPARTMENTS is VALID_DEPARTMENTS

    def test_loader_and_access_control_share_the_confidential_list(self):
        import src.ingestion.loader as loader

        assert loader._CONFIDENTIAL_DEPARTMENTS is CONFIDENTIAL_DEPARTMENTS

    def test_confidential_departments_are_real_departments(self):
        assert CONFIDENTIAL_DEPARTMENTS <= VALID_DEPARTMENTS


class TestKeyScopesContainer:
    def test_keys_excludes_scope_suffixes(self):
        assert parse_api_keys("a:hr,b:legal").keys == frozenset({"a", "b"})

    def test_unknown_key_has_no_scope(self):
        assert KeyScopes({"a": frozenset({"hr"})}).scope_for("missing") is None


# ---------------------------------------------------------------------------
# Enforcement through the HTTP layer
# ---------------------------------------------------------------------------

HR_KEY = "hr-key-aaaaaaaaaaaaaaaa"
ADMIN_KEY = "admin-key-bbbbbbbbbbbb"
API_KEYS = f"{ADMIN_KEY},{HR_KEY}:hr|general"


@pytest.fixture
def scoped_api():
    """API with one department-scoped key and one unscoped key."""
    from fastapi.testclient import TestClient

    s = MagicMock()
    s.validate = MagicMock()
    s.auth_enabled = True
    s.api_keys = API_KEYS
    s.chroma_collection = "test_collection"
    s.log_level = "WARNING"
    s.debug_mode = False
    s.max_upload_size_mb = 10
    s.cors_origins = "http://localhost:8501"
    s.cors_allow_methods = "GET,POST,OPTIONS"
    s.cors_allow_headers = "Content-Type,Authorization,X-Request-ID"
    s.ingest_root = "./data"
    s.async_ingestion = False
    s.guardrails_enabled = False
    s.rate_limit_storage_uri = ""

    with (
        patch("api.app.settings", s),
        patch("src.security.auth.settings", s),
    ):
        from api.app import app

        with TestClient(app) as tc:
            yield tc


def _auth(key):
    return {"Authorization": f"Bearer {key}"}


class TestAskEndpointScoping:
    def _capture_filter(self):
        """Patch the pipeline and record the filter it is handed."""
        from api.models import AskResponse

        seen = {}

        def fake_ask(body, session_id=None):
            seen["filter"] = body.filter
            return AskResponse(
                answer="ok", question=body.question, mode="naive",
                retriever_strategy="dense", cost_usd=0.0, latency_ms=1.0,
                tokens_used=0,
            )

        return seen, patch("api.app._ask_sync", side_effect=fake_ask)

    def test_scoped_key_asking_for_another_department_is_refused(self, scoped_api):
        seen, patched = self._capture_filter()
        with patched:
            resp = scoped_api.post(
                "/ask",
                json={"question": "severance terms?", "filter": {"department": "legal"}},
                headers=_auth(HR_KEY),
            )
        assert resp.status_code == 403
        assert "filter" not in seen, "pipeline must not run for a refused request"

    def test_the_refusal_does_not_name_what_it_is_hiding(self, scoped_api):
        seen, patched = self._capture_filter()
        with patched:
            body = scoped_api.post(
                "/ask",
                json={"question": "q", "filter": {"department": "security"}},
                headers=_auth(HR_KEY),
            ).text
        # It names the department the caller already supplied, but must
        # not enumerate the corpus or the caller's full grant.
        assert "test_collection" not in body

    def test_an_unfiltered_question_is_narrowed_to_the_caller_scope(self, scoped_api):
        """Without this the question searched every department."""
        seen, patched = self._capture_filter()
        with patched:
            resp = scoped_api.post(
                "/ask", json={"question": "what is the policy?"}, headers=_auth(HR_KEY)
            )
        assert resp.status_code == 200
        assert seen["filter"] == {"department": {"$in": ["general", "hr"]}}

    def test_a_permitted_department_is_still_honoured(self, scoped_api):
        seen, patched = self._capture_filter()
        with patched:
            resp = scoped_api.post(
                "/ask",
                json={"question": "leave policy?", "filter": {"department": "hr"}},
                headers=_auth(HR_KEY),
            )
        assert resp.status_code == 200
        assert seen["filter"] == {"department": "hr"}

    def test_an_unscoped_key_is_unaffected(self, scoped_api):
        """Legacy keys keep the behaviour they had before scoping existed."""
        seen, patched = self._capture_filter()
        with patched:
            resp = scoped_api.post(
                "/ask",
                json={"question": "q", "filter": {"department": "legal"}},
                headers=_auth(ADMIN_KEY),
            )
        assert resp.status_code == 200
        assert seen["filter"] == {"department": "legal"}

    def test_unscoped_key_with_no_filter_still_searches_everything(self, scoped_api):
        seen, patched = self._capture_filter()
        with patched:
            scoped_api.post("/ask", json={"question": "q"}, headers=_auth(ADMIN_KEY))
        assert seen["filter"] is None

    def test_streaming_requests_are_scoped_too(self, scoped_api):
        """The streaming path reads body.filter independently."""
        seen, patched = self._capture_filter()
        with patched:
            resp = scoped_api.post(
                "/ask",
                json={"question": "q", "stream": True, "filter": {"department": "legal"}},
                headers=_auth(HR_KEY),
            )
        assert resp.status_code == 403


class TestUploadScoping:
    """Write is the way around read scoping if it is not checked."""

    def _upload(self, client, key, department):
        return client.post(
            "/upload",
            files={"file": ("policy.txt", b"hello world", "text/plain")},
            data={"department": department},
            headers=_auth(key),
        )

    def test_scoped_key_cannot_file_into_another_department(self, scoped_api):
        assert self._upload(scoped_api, HR_KEY, "legal").status_code == 403

    def test_scoped_key_can_file_into_its_own_department(self, scoped_api):
        with patch("api.app._upload_sync") as up:
            from api.models import UploadResponse

            up.return_value = UploadResponse(
                filename="policy.txt", department="hr", documents_loaded=1,
                chunks_created=1, chunks_added=1, collection_total=1,
            )
            assert self._upload(scoped_api, HR_KEY, "hr").status_code == 200

    def test_unscoped_key_may_still_file_anywhere(self, scoped_api):
        with patch("api.app._upload_sync") as up:
            from api.models import UploadResponse

            up.return_value = UploadResponse(
                filename="policy.txt", department="legal", documents_loaded=1,
                chunks_created=1, chunks_added=1, collection_total=1,
            )
            assert self._upload(scoped_api, ADMIN_KEY, "legal").status_code == 200
