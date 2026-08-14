"""Central configuration. Loads from .env once and exposes typed settings."""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads .env in the project root

PROJECT_ROOT = Path(__file__).parent.resolve()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    chroma_dir: str = os.getenv("CHROMA_DIR", str(PROJECT_ROOT / "chroma_db"))
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "enterprise_docs")
    checkpoint_dir: str = os.getenv("CHECKPOINT_DIR", str(PROJECT_ROOT / "checkpoints"))
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    langsmith_api_key: str = os.getenv("LANGSMITH_API_KEY", "")
    langsmith_tracing: str = os.getenv("LANGSMITH_TRACING", "")
    langsmith_project: str = os.getenv("LANGSMITH_PROJECT", "enterprise-rag-assistant")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # API settings (Phase 7)
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))

    # Phase 9: Security
    cors_origins: str = os.getenv("CORS_ORIGINS", "http://localhost:8501")
    cors_allow_methods: str = os.getenv("CORS_ALLOW_METHODS", "GET,POST,OPTIONS")
    cors_allow_headers: str = os.getenv("CORS_ALLOW_HEADERS", "Content-Type,Authorization,X-Request-ID")
    debug_mode: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"
    max_upload_size_mb: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))

    # Retrieval defaults
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    top_k: int = int(os.getenv("TOP_K", "4"))

    # Phase 8: Resilience
    llm_timeout: int = int(os.getenv("LLM_TIMEOUT", "30"))
    llm_max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "2"))
    cost_alert_threshold: float = float(os.getenv("COST_ALERT_THRESHOLD", "0.05"))
    rerank_max_workers: int = int(os.getenv("RERANK_MAX_WORKERS", "4"))

    # Phase 8: Retrieval enhancements
    adaptive_k: bool = os.getenv("ADAPTIVE_K", "false").lower() == "true"
    adaptive_k_min: int = int(os.getenv("ADAPTIVE_K_MIN", "3"))
    adaptive_k_max: int = int(os.getenv("ADAPTIVE_K_MAX", "8"))
    per_doc_grading: bool = os.getenv("PER_DOC_GRADING", "false").lower() == "true"

    # Phase 8: Multi-part processing
    sub_query_max_retries: int = int(os.getenv("SUB_QUERY_MAX_RETRIES", "1"))
    parallel_sub_queries: bool = os.getenv("PARALLEL_SUB_QUERIES", "false").lower() == "true"
    sub_query_max_workers: int = int(os.getenv("SUB_QUERY_MAX_WORKERS", "3"))

    # Phase 8: Ingestion
    markdown_aware_chunking: bool = os.getenv("MARKDOWN_AWARE_CHUNKING", "true").lower() == "true"

    # Phase 8: Prompts & Tools
    chain_of_thought: bool = os.getenv("CHAIN_OF_THOUGHT", "false").lower() == "true"
    enable_tools: bool = os.getenv("ENABLE_TOOLS", "false").lower() == "true"

    # Phase 18: MCP Integration
    mcp_enabled: bool = os.getenv("MCP_ENABLED", "false").lower() == "true"

    # Phase 9: Architecture — centralized constants
    max_retries: int = int(os.getenv("MAX_RETRIES", "2"))
    max_sub_questions: int = int(os.getenv("MAX_SUB_QUESTIONS", "5"))
    num_query_variants: int = int(os.getenv("NUM_QUERY_VARIANTS", "3"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))
    rerank_fetch_k: int = int(os.getenv("RERANK_FETCH_K", "12"))
    rate_limit_per_minute: str = os.getenv("RATE_LIMIT_PER_MINUTE", "30/minute")
    heavy_rate_limit: str = os.getenv("HEAVY_RATE_LIMIT", "5/minute")
    cost_budget_per_query: float = float(os.getenv("COST_BUDGET_PER_QUERY", "0.02"))

    # Ingestion is restricted to paths under this root (prevents arbitrary
    # file reads via POST /ingest).
    ingest_root: str = os.getenv("INGEST_ROOT", str(PROJECT_ROOT / "data"))

    # Graph checkpointer backend: "memory" (default — request-scoped runs
    # never resume threads, so persistence is pure write overhead) or
    # "sqlite" for durable checkpoints.
    graph_checkpointer: str = os.getenv("GRAPH_CHECKPOINTER", "memory")

    # Phase 9: Circuit breaker
    circuit_breaker_threshold: int = int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5"))
    circuit_breaker_timeout: int = int(os.getenv("CIRCUIT_BREAKER_TIMEOUT", "60"))

    # Phase 17: Authentication & Guardrails
    auth_enabled: bool = os.getenv("AUTH_ENABLED", "false").lower() == "true"
    api_keys: str = os.getenv("API_KEYS", "")
    guardrails_enabled: bool = os.getenv("GUARDRAILS_ENABLED", "true").lower() == "true"
    max_query_length: int = int(os.getenv("MAX_QUERY_LENGTH", "2000"))
    pii_detection_enabled: bool = os.getenv("PII_DETECTION_ENABLED", "true").lower() == "true"

    # Phase 13: Cross-Encoder Reranker
    cross_encoder_model: str = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    cross_encoder_device: str = os.getenv("CROSS_ENCODER_DEVICE", "cpu")
    cross_encoder_batch_size: int = int(os.getenv("CROSS_ENCODER_BATCH_SIZE", "16"))

    # Phase 11: Intent Detection
    intent_detection_enabled: bool = os.getenv("INTENT_DETECTION_ENABLED", "true").lower() == "true"

    # Unified query analysis: one structured LLM call replaces the
    # intent_detect -> query_transform -> planner chain (3 calls),
    # cutting ~2-4s latency and ~30-40% of per-query LLM cost.
    unified_analysis: bool = os.getenv("UNIFIED_ANALYSIS", "false").lower() == "true"

    # Critic mode: "always" verifies every answer (default), "adaptive"
    # skips verification when the grader passed and the answer is short
    # (low hallucination risk), "off" disables the critic entirely.
    critic_mode: str = os.getenv("CRITIC_MODE", "always")
    critic_skip_max_chars: int = int(os.getenv("CRITIC_SKIP_MAX_CHARS", "600"))

    # Phase 12: Query Transformer
    query_transform_enabled: bool = os.getenv("QUERY_TRANSFORM_ENABLED", "true").lower() == "true"

    # Phase 14: Context Builder
    context_max_tokens: int = int(os.getenv("CONTEXT_MAX_TOKENS", "4000"))

    # Phase 16: Knowledge Graph
    knowledge_graph_enabled: bool = os.getenv("KNOWLEDGE_GRAPH_ENABLED", "false").lower() == "true"
    kg_max_depth: int = int(os.getenv("KG_MAX_DEPTH", "2"))
    kg_persist_path: str = os.getenv("KG_PERSIST_PATH", str(PROJECT_ROOT / "checkpoints" / "knowledge_graph.json"))

    # Phase 10: Conversation Memory
    memory_enabled: bool = os.getenv("MEMORY_ENABLED", "true").lower() == "true"
    memory_max_turns: int = int(os.getenv("MEMORY_MAX_TURNS", "10"))
    memory_max_tokens: int = int(os.getenv("MEMORY_MAX_TOKENS", "2000"))

    # -----------------------------------------------------------------
    # Phase 22: Event-driven ingestion (upload -> object store -> queue
    # -> worker). See ARCHITECTURE.md §1.2.
    # -----------------------------------------------------------------
    # When false (default), POST /upload keeps the legacy synchronous
    # behaviour: parse + embed inline, respond 200 with chunk counts.
    # When true, /upload durably stores the file, publishes an event and
    # responds 202 — a separate worker process does the indexing.
    async_ingestion: bool = os.getenv("ASYNC_INGESTION", "false").lower() == "true"

    # Object storage: "local" (filesystem, no extra deps) or "s3" (boto3;
    # also speaks to MinIO/any S3-compatible store via S3_ENDPOINT_URL).
    storage_backend: str = os.getenv("STORAGE_BACKEND", "local")
    storage_local_dir: str = os.getenv("STORAGE_LOCAL_DIR", str(PROJECT_ROOT / "object_store"))
    s3_bucket: str = os.getenv("S3_BUCKET", "")
    s3_region: str = os.getenv("S3_REGION", "us-east-1")
    s3_endpoint_url: str = os.getenv("S3_ENDPOINT_URL", "")
    s3_prefix: str = os.getenv("S3_PREFIX", "documents")

    # Event bus: "sqlite" (durable local queue with visibility timeout —
    # works with no broker running) or "kafka".
    event_bus: str = os.getenv("EVENT_BUS", "sqlite")
    event_bus_path: str = os.getenv(
        "EVENT_BUS_PATH", str(PROJECT_ROOT / "checkpoints" / "event_bus.db")
    )
    kafka_bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    kafka_topic_ingestion: str = os.getenv("KAFKA_TOPIC_INGESTION", "document.uploaded")
    kafka_topic_dlq: str = os.getenv("KAFKA_TOPIC_DLQ", "document.uploaded.dlq")
    kafka_consumer_group: str = os.getenv("KAFKA_CONSUMER_GROUP", "ingestion-workers")

    # Worker retry policy. The diagram calls for up to 3 attempts before
    # the event is routed to the dead-letter queue.
    ingest_max_attempts: int = int(os.getenv("INGEST_MAX_ATTEMPTS", "3"))
    ingest_retry_backoff_s: float = float(os.getenv("INGEST_RETRY_BACKOFF_S", "2.0"))
    # How long a claimed event stays invisible to other workers before it
    # is considered abandoned and redelivered (worker crash recovery).
    ingest_visibility_timeout_s: int = int(os.getenv("INGEST_VISIBILITY_TIMEOUT_S", "300"))
    worker_poll_interval_s: float = float(os.getenv("WORKER_POLL_INTERVAL_S", "1.0"))
    worker_batch_size: int = int(os.getenv("WORKER_BATCH_SIZE", "1"))
    document_registry_path: str = os.getenv(
        "DOCUMENT_REGISTRY_PATH", str(PROJECT_ROOT / "checkpoints" / "documents.db")
    )

    # Emit one JSON object per log line (for log aggregators). Includes the
    # request/trace id so an upload can be followed API -> queue -> worker.
    json_logs: bool = os.getenv("JSON_LOGS", "false").lower() == "true"

    # Phase 8: Infrastructure
    chroma_refresh_interval: int = int(os.getenv("CHROMA_REFRESH_INTERVAL", "300"))
    document_ttl_days: int = int(os.getenv("DOCUMENT_TTL_DAYS", "0"))
    semantic_cache_enabled: bool = os.getenv("SEMANTIC_CACHE_ENABLED", "false").lower() == "true"
    semantic_cache_threshold: float = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.95"))
    semantic_cache_ttl: int = int(os.getenv("SEMANTIC_CACHE_TTL", "3600"))

    _VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

    def validate(self) -> None:
        if not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        if self.log_level.upper() not in self._VALID_LOG_LEVELS:
            raise ValueError(
                f"Invalid LOG_LEVEL {self.log_level!r}. "
                f"Choose from: {', '.join(sorted(self._VALID_LOG_LEVELS))}"
            )
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than chunk_size ({self.chunk_size})"
            )
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")
        if self.llm_timeout <= 0:
            raise ValueError(f"llm_timeout must be positive, got {self.llm_timeout}")
        if self.llm_max_retries < 0:
            raise ValueError(f"llm_max_retries must be non-negative, got {self.llm_max_retries}")
        if self.cost_alert_threshold <= 0:
            raise ValueError(f"cost_alert_threshold must be positive, got {self.cost_alert_threshold}")
        if self.max_upload_size_mb <= 0:
            raise ValueError(f"max_upload_size_mb must be positive, got {self.max_upload_size_mb}")
        if self.critic_mode.lower() not in {"always", "adaptive", "off"}:
            raise ValueError(
                f"Invalid CRITIC_MODE {self.critic_mode!r}. Choose from: always, adaptive, off"
            )
        if self.storage_backend.lower() not in {"local", "s3"}:
            raise ValueError(
                f"Invalid STORAGE_BACKEND {self.storage_backend!r}. Choose from: local, s3"
            )
        if self.event_bus.lower() not in {"sqlite", "kafka"}:
            raise ValueError(
                f"Invalid EVENT_BUS {self.event_bus!r}. Choose from: sqlite, kafka"
            )
        if self.storage_backend.lower() == "s3" and not self.s3_bucket:
            raise ValueError("STORAGE_BACKEND=s3 requires S3_BUCKET to be set.")
        if self.ingest_max_attempts < 1:
            raise ValueError(
                f"ingest_max_attempts must be at least 1, got {self.ingest_max_attempts}"
            )
        if self.ingest_visibility_timeout_s <= 0:
            raise ValueError(
                f"ingest_visibility_timeout_s must be positive, got {self.ingest_visibility_timeout_s}"
            )
        if self.async_ingestion and self.event_bus.lower() == "sqlite":
            logging.getLogger(__name__).info(
                "ASYNC_INGESTION is on with the SQLite event bus — durable and "
                "correct for a single node, but workers must share the same "
                "EVENT_BUS_PATH. Use EVENT_BUS=kafka to scale out."
            )
        if self.langsmith_tracing.lower() == "true" and not self.langsmith_api_key:
            logging.getLogger(__name__).warning(
                "LANGSMITH_TRACING is enabled but LANGSMITH_API_KEY is not set — "
                "tracing will be silently skipped by LangSmith."
            )


settings = Settings()


class _JsonFormatter(logging.Formatter):
    """One JSON object per line, with the ambient request id attached.

    Log aggregators (CloudWatch, Loki, Datadog) index structured fields;
    the plain-text formatter forces them to regex the message instead.
    """

    def format(self, record: logging.LogRecord) -> str:
        import json

        from src.observability.request_context import get_request_id

        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = get_request_id()
        if request_id:
            payload["request_id"] = request_id
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # Anything attached via logger.info(..., extra={...}).
        for key, value in getattr(record, "__dict__", {}).items():
            if key.startswith("ctx_"):
                payload[key[4:]] = value
        return json.dumps(payload, default=str)


def setup_logging() -> None:
    """Configure logging for the entire application. Call once at entry point."""
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    if settings.json_logs:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    root = logging.getLogger()
    root.setLevel(level)
    # Avoid adding duplicate handlers on repeated calls.
    if not root.handlers:
        root.addHandler(handler)
    # Quiet noisy third-party loggers.
    for noisy in ("httpx", "httpcore", "chromadb", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
