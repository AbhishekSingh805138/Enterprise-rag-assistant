"""Load raw documents from a directory into LangChain Document objects.

Supports .pdf, .docx, .csv, .txt and .md. Each loaded Document carries
metadata (source path, file type, department, access_level) so the vector
store can do metadata-filtered retrieval later.

.docx needs the optional ``docx2txt`` package; the failure is reported per
file and skipped rather than aborting a whole directory ingest.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".csv", ".txt", ".md"}

# File types whose loader needs a package that is not guaranteed to be
# installed. Declaring a type here means the upload API will refuse it
# with a clear message when the dependency is absent, rather than
# accepting the file and dead-lettering it seconds later.
OPTIONAL_DEPENDENCIES = {
    ".pdf": "pypdf",
    ".docx": "docx2txt",
}

_availability_cache: dict[str, bool] = {}


def _dependency_installed(module: str) -> bool:
    """Whether *module* can be imported, without importing it."""
    if module not in _availability_cache:
        import importlib.util

        try:
            _availability_cache[module] = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            _availability_cache[module] = False
    return _availability_cache[module]


def missing_dependency(suffix: str) -> str | None:
    """Return the package needed to load *suffix*, or None if it is ready."""
    module = OPTIONAL_DEPENDENCIES.get(suffix.lower())
    if module and not _dependency_installed(module):
        return module
    return None


def available_suffixes() -> frozenset[str]:
    """Suffixes this deployment can actually parse right now.

    Distinct from SUPPORTED_SUFFIXES, which is what the code knows how to
    handle in principle. Validation must use this one: promising a
    capability the runtime lacks turns a clean 400 into an accepted upload
    that fails asynchronously, where the user never sees the reason.
    """
    return frozenset(s for s in SUPPORTED_SUFFIXES if missing_dependency(s) is None)


def reset_availability_cache() -> None:
    """Clear the cached dependency probes (tests, or after an install)."""
    _availability_cache.clear()


def _load_docx(path: Path) -> list[Document]:
    """Load a .docx via docx2txt, with an actionable error when it is absent."""
    try:
        import docx2txt  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Loading .docx files requires docx2txt. Install it with: pip install docx2txt"
        ) from e
    from langchain_community.document_loaders import Docx2txtLoader

    return Docx2txtLoader(str(path)).load()


def _load_csv(path: Path) -> list[Document]:
    """Load a .csv as one Document per row (LangChain's CSVLoader default).

    Row-level granularity keeps each embedded unit semantically whole; the
    chunker leaves them alone since rows are almost always under the chunk
    size.
    """
    from langchain_community.document_loaders import CSVLoader

    return CSVLoader(str(path), encoding="utf-8").load()

# Documents in these folders are marked confidential; everything else is
# internal. Shared with the retrieval scoping rules so "confidential"
# means the same thing at write time and at read time.
from src.security.access_control import CONFIDENTIAL_DEPARTMENTS as _CONFIDENTIAL_DEPARTMENTS


def _infer_department(file_path: Path, root: Path) -> str:
    """Derive department from the first subfolder under *root*.

    Example: root/legal/contract.md  →  "legal"
             root/report.md          →  "general"
    """
    try:
        relative = file_path.relative_to(root)
        parts = relative.parts
        if len(parts) > 1:
            return parts[0].lower()
    except ValueError:
        pass
    return "general"


def _infer_access_level(department: str) -> str:
    """Simple rule: confidential departments get 'confidential', rest 'internal'."""
    return "confidential" if department in _CONFIDENTIAL_DEPARTMENTS else "internal"


def load_path(path: str | Path) -> list[Document]:
    """Load a single file or every supported file in a directory.

    Enriches each Document's metadata with:
      - source, filename, doc_type  (original)
      - department, access_level    (new — inferred from folder structure)
    """
    path = Path(path).resolve()
    root = path if path.is_dir() else path.parent

    files = (
        [path]
        if path.is_file()
        else sorted(p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
    )

    if not files:
        raise FileNotFoundError(f"No supported documents found under {path!s}")

    docs: list[Document] = []
    for f in files:
        suffix = f.suffix.lower()
        try:
            if suffix == ".pdf":
                loaded = PyPDFLoader(str(f)).load()
            elif suffix == ".docx":
                loaded = _load_docx(f)
            elif suffix == ".csv":
                loaded = _load_csv(f)
            elif suffix in {".txt", ".md"}:
                loaded = TextLoader(str(f), encoding="utf-8").load()
            else:
                continue
        except Exception:
            logger.exception("Failed to load %s — skipping", f)
            continue

        department = _infer_department(f, root)
        access_level = _infer_access_level(department)

        for d in loaded:
            d.metadata.setdefault("source", str(f))
            d.metadata["doc_type"] = suffix.lstrip(".")
            d.metadata["filename"] = f.name
            d.metadata["department"] = department
            d.metadata["access_level"] = access_level
            d.metadata["ingested_at"] = datetime.now(UTC).isoformat()
        docs.extend(loaded)

        logger.info(
            "Loaded %s (%d page(s), dept=%s, access=%s)",
            f.name, len(loaded), department, access_level,
        )

    if not docs:
        raise FileNotFoundError(f"No supported documents found under {path!s}")

    logger.info("Total documents loaded: %d from %d file(s)", len(docs), len(files))
    return docs
