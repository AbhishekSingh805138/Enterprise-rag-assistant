"""Tests for the document loader."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion.loader import _infer_access_level, _infer_department, load_path


class TestInferDepartment:
    def test_subfolder_detected(self, tmp_path):
        f = tmp_path / "legal" / "contract.md"
        f.parent.mkdir()
        f.touch()
        assert _infer_department(f, tmp_path) == "legal"

    def test_nested_subfolder_uses_first_level(self, tmp_path):
        f = tmp_path / "engineering" / "runbooks" / "deploy.md"
        f.parent.mkdir(parents=True)
        f.touch()
        assert _infer_department(f, tmp_path) == "engineering"

    def test_root_file_returns_general(self, tmp_path):
        f = tmp_path / "readme.md"
        f.touch()
        assert _infer_department(f, tmp_path) == "general"

    def test_case_insensitive(self, tmp_path):
        f = tmp_path / "HR" / "policy.md"
        f.parent.mkdir()
        f.touch()
        # The department is lowercased
        assert _infer_department(f, tmp_path) == "hr"


class TestInferAccessLevel:
    def test_legal_is_confidential(self):
        assert _infer_access_level("legal") == "confidential"

    def test_security_is_confidential(self):
        assert _infer_access_level("security") == "confidential"

    def test_hr_is_internal(self):
        assert _infer_access_level("hr") == "internal"

    def test_engineering_is_internal(self):
        assert _infer_access_level("engineering") == "internal"

    def test_general_is_internal(self):
        assert _infer_access_level("general") == "internal"


class TestLoadPath:
    def test_loads_directory(self, sample_docs_path):
        docs = load_path(sample_docs_path)
        assert len(docs) >= 3  # at least the 3 files we created

    def test_metadata_enrichment(self, sample_docs_path):
        docs = load_path(sample_docs_path)
        for doc in docs:
            assert "source" in doc.metadata
            assert "filename" in doc.metadata
            assert "doc_type" in doc.metadata
            assert "department" in doc.metadata
            assert "access_level" in doc.metadata

    def test_department_from_folder(self, sample_docs_path):
        docs = load_path(sample_docs_path)
        departments = {d.metadata["department"] for d in docs}
        assert "hr" in departments
        assert "legal" in departments

    def test_access_level_set(self, sample_docs_path):
        docs = load_path(sample_docs_path)
        legal_docs = [d for d in docs if d.metadata["department"] == "legal"]
        assert all(d.metadata["access_level"] == "confidential" for d in legal_docs)
        hr_docs = [d for d in docs if d.metadata["department"] == "hr"]
        assert all(d.metadata["access_level"] == "internal" for d in hr_docs)

    def test_single_file(self, sample_docs_path):
        single = Path(sample_docs_path) / "hr" / "policy.md"
        docs = load_path(str(single))
        assert len(docs) == 1
        assert docs[0].metadata["filename"] == "policy.md"

    def test_empty_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_path(str(tmp_path))

    def test_unsupported_files_skipped(self, tmp_path):
        # .xlsx/.jpg are outside SUPPORTED_SUFFIXES, so nothing loads.
        (tmp_path / "sheet.xlsx").write_bytes(b"PK\x03\x04fake")
        (tmp_path / "photo.jpg").write_bytes(b"\xff\xd8\xff fake")
        with pytest.raises(FileNotFoundError):
            load_path(str(tmp_path))

    def test_csv_is_supported(self, tmp_path):
        """CSV is one of the ingestion architecture's accepted types."""
        (tmp_path / "data.csv").write_text("name,role\nada,engineer\n", encoding="utf-8")
        docs = load_path(str(tmp_path))
        assert len(docs) == 1
        assert docs[0].metadata["doc_type"] == "csv"
        assert "ada" in docs[0].page_content

    def test_unsupported_files_are_ignored_alongside_supported_ones(self, tmp_path):
        (tmp_path / "notes.md").write_text("# Notes\nHello.\n", encoding="utf-8")
        (tmp_path / "photo.jpg").write_bytes(b"\xff\xd8\xff fake")
        docs = load_path(str(tmp_path))
        assert [d.metadata["filename"] for d in docs] == ["notes.md"]

    def test_doc_type_correct(self, sample_docs_path):
        docs = load_path(sample_docs_path)
        for doc in docs:
            assert doc.metadata["doc_type"] in ("md", "txt", "pdf")
