"""Pydantic request/response models for the API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.security.access_control import VALID_DEPARTMENTS as ACCESS_CONTROL_DEPARTMENTS

# Allowed department names for upload and data lookup. Re-exported from
# the access-control module so the API and the retrieval scoping rules
# can never disagree about which departments exist — a name valid in one
# place but not the other would be either an upload that cannot be read
# back, or a scope that silently grants nothing.
VALID_DEPARTMENTS = ACCESS_CONTROL_DEPARTMENTS


# ---------------------------------------------------------------------------
# /ask
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask")
    mode: Literal["naive", "graph", "auto"] = "naive"
    retriever_strategy: Literal[
        "dense", "hybrid", "multi_query", "rerank", "hybrid_rerank",
        "cross_rerank", "hybrid_cross_rerank", "knowledge_graph", "auto",
    ] = "dense"
    filter: dict[str, str] | None = None
    top_k: int | None = Field(None, gt=0)
    stream: bool = False
    session_id: str | None = Field(None, description="Session ID for multi-turn conversations")


class AskResponse(BaseModel):
    answer: str
    question: str
    mode: str
    retriever_strategy: str
    cost_usd: float
    latency_ms: float
    tokens_used: int
    node_latencies: dict[str, float] | None = None
    is_idk: bool = False
    session_id: str | None = None
    intent: str | None = None
    cache_hit: bool = False


# ---------------------------------------------------------------------------
# /ingest
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    path: str = "./data/sample_docs"
    chunk_size: int | None = Field(None, gt=0)
    chunk_overlap: int | None = Field(None, ge=0)


class IngestResponse(BaseModel):
    documents_loaded: int
    chunks_created: int
    chunks_added: int
    collection_total: int


# ---------------------------------------------------------------------------
# /upload
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    """Synchronous upload result (ASYNC_INGESTION=false)."""

    filename: str
    documents_loaded: int
    chunks_created: int
    chunks_added: int
    collection_total: int


class UploadAcceptedResponse(BaseModel):
    """202 result for the async path: the file is stored and queued.

    Indexing happens on a worker, so poll GET /documents/{document_id}
    until status is PROCESSED (or DEAD_LETTER).
    """

    document_id: str
    filename: str
    department: str
    status: str
    duplicate: bool = Field(
        False,
        description="True when this content was already registered; no new work was queued.",
    )
    checksum: str
    size_bytes: int
    storage_uri: str = ""
    status_url: str = ""
    message: str = ""


# ---------------------------------------------------------------------------
# /documents
# ---------------------------------------------------------------------------

class DocumentStatusResponse(BaseModel):
    """Lifecycle state of one registered document."""

    document_id: str
    filename: str
    department: str
    status: Literal["PENDING", "PROCESSING", "PROCESSED", "FAILED", "DEAD_LETTER"]
    attempts: int
    chunks_indexed: int
    size_bytes: int
    checksum: str
    content_type: str = ""
    storage_uri: str = ""
    error: str = ""
    created_at: str = ""
    updated_at: str = ""


class DocumentListResponse(BaseModel):
    documents: list[DocumentStatusResponse]
    count: int
    stats: dict[str, int]


class DocumentRetryResponse(BaseModel):
    """Result of requeueing a failed or dead-lettered document."""

    document_id: str
    filename: str = ""
    outcome: str = Field(
        ..., description="requeued | already_processed | not_replayable | publish_failed"
    )
    detail: str = ""
    requeued: bool = False
    status_url: str = ""


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    collection: str
    document_count: int
    version: str


# ---------------------------------------------------------------------------
# /eval
# ---------------------------------------------------------------------------

class EvalRequest(BaseModel):
    mode: Literal["naive", "graph", "auto"] = "naive"
    retriever_strategy: Literal[
        "dense", "hybrid", "multi_query", "rerank", "hybrid_rerank",
        "cross_rerank", "hybrid_cross_rerank", "knowledge_graph", "auto",
    ] = "dense"
    limit: int | None = Field(None, gt=0)


class EvalResponse(BaseModel):
    scores: dict[str, float]
    items_evaluated: int
    mode: str
    retriever_strategy: str
    duration_s: float


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    error: str
    detail: str
