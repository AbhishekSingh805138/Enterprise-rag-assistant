# Enterprise RAG Assistant

A production-grade **Retrieval-Augmented Generation (RAG)** system built for enterprise document intelligence. Features a Corrective RAG (CRAG) pipeline with LangGraph orchestration, 8 retrieval strategies, multi-agent question decomposition, conversation memory, knowledge graph, an event-driven ingestion pipeline, and a full security layer.

**Tech Stack**: LangChain 1.0 | LangGraph 1.0 | ChromaDB | OpenAI | FastAPI | Streamlit | Kafka | S3

---

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Pipeline Flow](#pipeline-flow)
- [Retrieval Strategies](#retrieval-strategies)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Docker Deployment](#docker-deployment)
- [API Reference](#api-reference)
- [CLI Tools](#cli-tools)
- [Async Ingestion](#async-ingestion)
- [Testing](#testing)
- [Evaluation](#evaluation)
- [Security](#security)
- [Observability](#observability)
- [Sample Data](#sample-data)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features

### Core RAG Pipeline
- **Corrective RAG (CRAG)** graph with retrieve -> grade -> rewrite/retry -> web fallback -> generate loop
- **8 retrieval strategies**: dense, hybrid (BM25+vector), multi-query expansion, LLM reranking, cross-encoder reranking, hybrid+rerank combos, knowledge graph
- **Multi-agent decomposition**: LLM planner splits complex questions into sub-queries, processes in parallel, synthesizes final answer
- **Reciprocal Rank Fusion (RRF)**: Combines sparse and dense retrieval scores with configurable k parameter
- **Cross-encoder reranking**: Sentence-transformer model (`ms-marco-MiniLM-L-6-v2`) for two-stage ranking

### Intelligence Layer
- **Intent detection**: Classifies queries (informational, comparative, procedural, analytical, multi-hop, factual) for downstream routing
- **Query transformation**: Entity extraction, synonym injection, intent-aware rewriting
- **Scope detection**: Keyword + regex heuristics reject out-of-domain queries without an LLM call
- **Knowledge graph**: NetworkX-backed entity-relationship graph with multi-hop traversal

### Conversation & Caching
- **Multi-turn memory**: SQLite-backed conversation history with session management and token-budgeted context
- **Semantic cache**: Cosine similarity matching (threshold 0.95) for instant responses on repeated queries

### Ingestion Pipeline
- **Event-driven ingestion**: `POST /upload` stores the file durably, publishes an event and returns `202` — parsing and embedding happen on a separate worker, off the request path
- **Content-addressed idempotency**: `sha256(bytes)` + department derives the document ID, so a repeat upload is rejected before any parse or embed cost is paid
- **Retry with backoff, then DLQ**: transient failures retried up to 3 times; permanent ones (corrupt file, missing object) dead-lettered immediately instead of burning the budget
- **Crash recovery**: visibility timeouts redeliver events abandoned by a killed worker; stale `PROCESSING` claims are reclaimable so documents are never stranded
- **Pluggable backends**: object storage (`local` / `s3`+MinIO) and event bus (`sqlite` / `kafka`) — the defaults need no broker, so CI exercises the same retry and DLQ code production runs
- **Lifecycle visibility**: `GET /documents/{id}` reports `PENDING → PROCESSING → PROCESSED / FAILED / DEAD_LETTER`
- **File types**: PDF, DOCX, CSV, TXT, Markdown

### Security & Guardrails
- **API authentication**: Static API keys or JWT token validation
- **Input guardrails**: Prompt injection detection (11 regex patterns), PII detection (SSN, credit card, phone, email), max query length enforcement
- **Output filtering**: PII redaction in LLM responses before returning to client
- **Rate limiting**: Configurable per-minute request limits via slowapi

### Resilience
- **Circuit breaker**: Exponential backoff with CLOSED/OPEN/HALF_OPEN state transitions
- **Configurable timeouts & retries**: Per-LLM-call timeout with retry budgets
- **Thread-safe singletons**: All stores use lock-protected singleton pattern

### Observability
- **Cost & token tracking**: Per-query cost calculation via LangChain callbacks
- **Latency metrics**: Per-node timing with SQLite persistence
- **Health checks**: Deep subsystem checks (ChromaDB, SQLite, object store, queue backlog, DLQ depth)
- **Request tracing**: `X-Request-ID` propagated from the API through the queue into the worker
- **Structured logging**: `JSON_LOGS=true` emits one JSON object per line with the request ID attached
- **LangSmith integration**: Optional distributed tracing

### API & UI
- **FastAPI REST API**: 8 endpoints with SSE streaming, CORS, rate limiting
- **Streamlit chat UI**: Dark theme, session persistence, pipeline progress visualization, file upload
- **Docker support**: Multi-service compose with health checks; production topology adds Kafka, MinIO and scalable ingestion workers

---

## Architecture Overview

The system has two independent paths: a **query path** that answers questions,
and an **ingestion path** that gets documents into the vector store. They meet
only at ChromaDB, so indexing a 200-page PDF never slows down a chat request.

```
+------------------+     +---------------------------+     +----------------+
| Streamlit UI     |     | FastAPI API Layer          |     | OpenAI API     |
| - Chat interface |---->| - POST /ask (SSE stream)   |---->| gpt-4o-mini    |
| - File upload    |     | - POST /upload      (202)  |     | text-embedding |
| - Session mgmt   |     | - GET  /documents/{id}     |     | -3-small       |
+------------------+     | - POST /ingest             |     +----------------+
                         | - POST /eval               |     | Tavily API     |
                         | - GET  /health /tools      |     | (web fallback) |
                         +-------------+-------------+     +----------------+
                                       |
              +------------------------+------------------------+
              |                                                 |
        QUERY PATH  /ask                              INGESTION PATH  /upload
              |                                                 |
              v                                                 v
    Security Layer + Rate Limiting                 Validate -> checksum ->
    (Auth, Guardrails, PII, 30/min)                idempotency check
              |                                                 |
              v                                    +------------+------------+
    LangGraph CRAG Pipeline                        | exists? -> 200 + same   |
    (see Pipeline Flow below)                      |            document_id  |
              |                                    +------------+------------+
              |                                                 | new
              |                                                 v
              |                                    Object Store (S3 / local)
              |                                                 |
              |                                                 v
              |                                    Event Queue (Kafka / SQLite)
              |                                       topic: document.uploaded
              |                                                 |
              |                                                 v
              |                                    Ingestion Worker(s)  x N
              |                                    parse -> chunk -> embed
              |                                    retry x3 -> DLQ on failure
              |                                                 |
              +--------------------+----------------------------+
                                   v
          +---------+---------+-----------+------------------+
          |         |         |           |                  |
       ChromaDB  SQLite    NetworkX   BM25Okapi        Document Registry
       (vectors) (memory,  (knowledge  (sparse         (SQLite: status,
                  metrics,  graph)     retrieval)       attempts, checksum)
                  cache)
```

Ingestion is synchronous by default (`ASYNC_INGESTION=false`) — the pipeline
above activates when you turn it on. See [Async Ingestion](#async-ingestion).

---

## Pipeline Flow

The CRAG pipeline is a stateful, cyclic graph built with LangGraph's `StateGraph`:

```
START
  |
  v
[guardrail_check] --blocked--> rejection message --> END
  | pass
  v
[load_memory] -- loads conversation history from SQLite
  |
  v
[cache_lookup] --cache hit--> [save_memory] --> END
  | cache miss
  v
[scope_check] --out of scope--> "I don't know" --> END
  | in scope
  v
[intent_detect] -- classifies query intent (6 types)
  |
  v
[query_transform] -- normalize, extract entities, rewrite
  |
  v
[tool_router] -- calculator / data_lookup / MCP tools
  |
  v
[planner] -- decompose into sub-questions if complex
  |
  |   (with UNIFIED_ANALYSIS=true, the three nodes intent_detect +
  |    query_transform + planner collapse into a single analyze_query
  |    node — one structured LLM call instead of three. See note below.)
  |
  +-- simple query --------+-- multi-part query ------+
  |                         |                          |
  v                         v                          |
[retrieve]            [process_sub_queries]             |
  |                   (sequential or parallel)          |
  v                         |                          |
[grade_documents]           v                          |
  |                   [synthesize]                     |
  +-- relevant --------> [generate]                    |
  |                         |                          |
  +-- retry (< 2) -----> [transform_query] --> [retrieve]
  |
  +-- exhausted (>= 2) -> [web_search] --> [generate]
                                              |
                                              v
                                          [critic] -- extract claims, verify
                                              |       vs sources, strip unsupported
                                              v
                                        [cache_store] -- saves for future hits
                                              |
                                              v
                                        [save_memory] -- persists conversation
                                              |
                                              v
                                             END
```

### Latency / cost optimizations (opt-in)

The full path can issue 6–8 serial LLM calls. Two feature flags cut that
down without changing default behavior (both default off):

| Flag | Effect |
|------|--------|
| `UNIFIED_ANALYSIS=true` | Replaces `intent_detect` → `query_transform` → `planner` with a single `analyze_query` node (intent + entities + query rewrite + decomposition in one structured LLM call), removing two serial round-trips. Falls back to zero-LLM heuristics on error/circuit-open. |
| `CRITIC_MODE` | `always` (default) verifies every answer; `adaptive` skips verification for low-risk answers (grader passed, no web fallback, not multi-part, under `CRITIC_SKIP_MAX_CHARS`); `off` disables the critic. |

Retrieval also supports `retriever_strategy=auto`, which routes by detected
intent: `factual` → `dense`, `comparative`/`analytical` → `hybrid_cross_rerank`,
everything else → `hybrid`.

---

## Retrieval Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `dense` | ChromaDB vector similarity search | General queries |
| `hybrid` | Dense + BM25 sparse retrieval fused via RRF | Keyword-heavy queries |
| `multi_query` | LLM generates 3 query variants, retrieves all, deduplicates | Ambiguous queries |
| `rerank` | Dense retrieval + LLM-based relevance scoring | High precision needs |
| `hybrid_rerank` | Hybrid retrieval + LLM reranking | Best of both worlds |
| `cross_rerank` | Dense + cross-encoder model reranking | Fast, high-quality reranking |
| `hybrid_cross_rerank` | Hybrid + cross-encoder reranking | Maximum retrieval quality |
| `knowledge_graph` | Entity extraction + graph traversal + source retrieval | Multi-hop reasoning |
| `auto` | Resolves to a concrete strategy from the query's detected intent (factual→dense, comparative/analytical→hybrid_cross_rerank, else hybrid) | Hands-off; let intent pick |

---

## Project Structure

```
enterprise-rag-assistant/
├── api/
│   ├── app.py                 # FastAPI application (6 endpoints + middleware)
│   └── models.py              # Pydantic request/response schemas
│
├── src/
│   ├── cache/
│   │   └── semantic_cache.py  # Embedding-based query cache (SQLite)
│   ├── context/
│   │   └── context_builder.py # Deduplication, grouping, token budgeting
│   ├── eval/
│   │   ├── ragas_eval.py      # RAGAS evaluation harness
│   │   └── eval_set.json      # Ground-truth evaluation dataset
│   ├── graph/
│   │   ├── state.py           # LangGraph shared state (TypedDict)
│   │   ├── build_graph.py     # CRAG graph compilation & routing
│   │   ├── nodes.py           # Core nodes: retrieve, grade, generate, critic
│   │   ├── analyzer.py        # Unified query analysis (UNIFIED_ANALYSIS)
│   │   ├── cache_nodes.py     # Cache lookup/store nodes
│   │   ├── guardrail_node.py  # Input safety check node
│   │   ├── intent_detector.py # Query intent classification
│   │   ├── memory_nodes.py    # Conversation memory load/save
│   │   ├── planner.py         # Multi-part question decomposition
│   │   ├── scope_detector.py  # Domain scope detection (44 keywords)
│   │   ├── tool_node.py       # Tool routing (MCP + regex fallback)
│   │   └── tracing.py         # Per-node performance tracing
│   ├── ingestion/
│   │   ├── loader.py          # PDF/DOCX/CSV/TXT/MD loaders with metadata
│   │   ├── chunker.py         # Recursive + markdown-aware splitting
│   │   ├── registry.py        # Document lifecycle + idempotency (SQLite)
│   │   ├── pipeline.py        # download -> parse -> chunk -> embed -> store
│   │   └── worker.py          # Queue consumer: retry x3, DLQ, crash recovery
│   ├── storage/
│   │   └── object_store.py    # Durable blob storage (local filesystem / S3)
│   ├── events/
│   │   ├── bus.py             # Event envelope + bus protocol + factory
│   │   ├── sqlite_bus.py      # Durable queue with visibility timeouts
│   │   └── kafka_bus.py       # Kafka producer/consumer, manual offset commit
│   ├── knowledge_graph/
│   │   ├── models.py          # Entity, Relationship, Triple models
│   │   ├── extractor.py       # LLM entity-relationship extraction
│   │   ├── retriever.py       # Graph-based document retrieval
│   │   └── store.py           # NetworkX graph with JSON persistence
│   ├── mcp/
│   │   ├── tool_registry.py   # MCP tool metadata registry
│   │   └── tool_router.py     # LLM function-calling tool selection
│   ├── memory/
│   │   ├── conversation_store.py  # SQLite session store
│   │   └── context_builder.py     # Token-budgeted history formatting
│   ├── observability/
│   │   ├── cost_callback.py   # Per-query cost/token tracking
│   │   ├── health_checker.py  # Subsystem health checks (incl. queue/DLQ depth)
│   │   ├── request_context.py # X-Request-ID propagation across processes
│   │   └── metrics_store.py   # SQLite metrics persistence
│   ├── rag/
│   │   └── naive_rag.py       # Baseline LCEL chain (no graph)
│   ├── resilience/
│   │   └── circuit_breaker.py # Circuit breaker with exponential backoff
│   ├── retrieval/
│   │   ├── factory.py         # Strategy factory (8 strategies)
│   │   ├── composed.py        # Chained retriever + reranker
│   │   ├── cross_encoder_rerank.py  # Sentence-transformer reranking
│   │   ├── hybrid.py          # Dense + BM25 with RRF fusion
│   │   ├── multi_query.py     # LLM query expansion
│   │   ├── rerank.py          # LLM-based reranking
│   │   ├── query_transformer.py     # Unified query transformation
│   │   ├── entity_extractor.py      # Named entity extraction
│   │   ├── dept_detector.py         # Department detection
│   │   └── normalizer.py           # Query normalization
│   ├── security/
│   │   ├── auth.py            # API key / JWT authentication
│   │   ├── guardrails.py      # Input validation (injection, PII, length)
│   │   └── output_filter.py   # PII redaction in responses
│   ├── tools/
│   │   ├── calculator.py      # Safe arithmetic evaluation
│   │   └── data_lookup.py     # Department-filtered document lookup
│   └── vectorstore/
│       └── chroma_store.py    # ChromaDB wrapper (OpenAI embeddings)
│
├── ui/
│   └── app.py                 # Streamlit chat interface
│
├── scripts/
│   ├── ask.py                 # CLI query tool (--mode graph/naive/auto)
│   ├── ingest.py              # Batch document ingestion
│   ├── worker.py              # Ingestion worker entrypoint (graceful shutdown)
│   ├── cleanup_stale.py       # Document TTL cleanup
│   ├── metrics.py             # Metrics dashboard CLI
│   └── upload_eval_dataset.py # Evaluation dataset uploader
│
├── tests/                     # 1234 tests across 52 files (93% coverage)
│   ├── conftest.py            # Shared fixtures
│   └── test_*.py              # Unit, integration, and e2e tests
│
├── data/sample_docs/          # Sample enterprise documents (6 departments)
├── Dockerfile                 # Python 3.11-slim container
├── docker-compose.yml         # Dev: API + UI (+ worker via --profile async)
├── docker-compose.prod.yml    # Prod: API + workers + Kafka + MinIO + UI
├── requirements.txt           # Python dependencies
├── .env.example               # Configuration template
├── .gitignore                 # Secrets, caches, build artifacts excluded
├── pytest.ini                 # Test configuration
└── ARCHITECTURE.md            # Detailed architecture documentation
```

---

## Prerequisites

- **Python 3.11+**
- **OpenAI API key** (for `gpt-4o-mini` and `text-embedding-3-small`)
- **Tavily API key** (optional, for web search fallback)
- **Docker & Docker Compose** (optional, for containerized deployment)
- **Kafka + S3/MinIO** (optional — only for `EVENT_BUS=kafka` / `STORAGE_BACKEND=s3`; the async ingestion defaults run entirely on local files, no broker needed)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/enterprise-rag-assistant.git
cd enterprise-rag-assistant
```

### 2. Create and activate virtual environment

```bash
# Linux/macOS
python -m venv .venv && source .venv/bin/activate

# Windows
python -m venv .venv && .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and add your API key:

```
OPENAI_API_KEY=sk-your-key-here
```

### 5. Ingest sample documents

```bash
python -m scripts.ingest ./data/sample_docs
```

---

## Configuration

All settings are managed via environment variables (loaded from `.env`). The system uses a frozen dataclass with sensible defaults -- you only need to set what you want to change.

### Required

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Your OpenAI API key |

### Models

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `gpt-4o-mini` | LLM for generation, grading, planning |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model for vector search |

### Retrieval

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | `1000` | Document chunk size (characters) |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K` | `4` | Number of documents to retrieve |
| `RRF_K` | `60` | RRF fusion parameter |
| `CROSS_ENCODER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model |
| `CROSS_ENCODER_DEVICE` | `cpu` | Device for cross-encoder (`cpu` or `cuda`) |

### Multi-Part Questions

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_SUB_QUESTIONS` | `5` | Maximum sub-questions for decomposition |
| `PARALLEL_SUB_QUERIES` | `false` | Process sub-queries in parallel |
| `SUB_QUERY_MAX_WORKERS` | `3` | Thread pool size for parallel processing |

### Pipeline Optimization

| Variable | Default | Description |
|----------|---------|-------------|
| `UNIFIED_ANALYSIS` | `false` | Collapse intent + query rewrite + decomposition into one LLM call (`analyze_query`) |
| `CRITIC_MODE` | `always` | Answer verification: `always`, `adaptive` (skip low-risk), or `off` |
| `CRITIC_SKIP_MAX_CHARS` | `600` | Under `adaptive`, max answer length eligible to skip the critic |
| `INTENT_DETECTION_ENABLED` | `true` | Enable intent classification node |
| `QUERY_TRANSFORM_ENABLED` | `true` | Enable query normalization/rewrite node |

### Memory & Caching

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY_ENABLED` | `true` | Enable conversation memory |
| `MEMORY_MAX_TURNS` | `10` | Maximum conversation turns to retain |
| `MEMORY_MAX_TOKENS` | `2000` | Token budget for memory context |
| `SEMANTIC_CACHE_ENABLED` | `false` | Enable semantic query cache |
| `SEMANTIC_CACHE_THRESHOLD` | `0.95` | Cosine similarity threshold for cache hit |
| `SEMANTIC_CACHE_TTL` | `3600` | Cache entry TTL (seconds) |

### Security

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTH_ENABLED` | `false` | Enable API key authentication |
| `API_KEYS` | `""` | Comma-separated valid API keys |
| `GUARDRAILS_ENABLED` | `true` | Enable input guardrails |
| `MAX_QUERY_LENGTH` | `2000` | Maximum allowed query length |
| `PII_DETECTION_ENABLED` | `true` | Enable PII detection & redaction |

### Resilience

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_TIMEOUT` | `30` | LLM call timeout (seconds) |
| `LLM_MAX_RETRIES` | `2` | Maximum LLM retries |
| `CIRCUIT_BREAKER_THRESHOLD` | `5` | Failures before circuit opens |
| `CIRCUIT_BREAKER_TIMEOUT` | `60` | Seconds before half-open |

### Observability

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Log level |
| `LANGSMITH_TRACING` | `""` | Enable LangSmith tracing (`true`/`false`) |
| `LANGSMITH_PROJECT` | `enterprise-rag-assistant` | LangSmith project name |

### Knowledge Graph

| Variable | Default | Description |
|----------|---------|-------------|
| `KNOWLEDGE_GRAPH_ENABLED` | `false` | Enable knowledge graph retrieval |
| `KG_MAX_DEPTH` | `2` | Graph traversal depth |

### Infrastructure

| Variable | Default | Description |
|----------|---------|-------------|
| `CHROMA_DIR` | `./chroma_db` | ChromaDB persistence directory |
| `CHROMA_COLLECTION` | `enterprise_docs` | Collection name |
| `CHECKPOINT_DIR` | `./checkpoints` | State persistence directory |
| `CORS_ORIGINS` | `http://localhost:8501` | Allowed CORS origins |
| `MAX_UPLOAD_SIZE_MB` | `10` | Maximum file upload size |

### Async Ingestion

| Variable | Default | Description |
|----------|---------|-------------|
| `ASYNC_INGESTION` | `false` | Store + queue uploads and return `202` instead of indexing inline |
| `STORAGE_BACKEND` | `local` | Object storage: `local` or `s3` |
| `STORAGE_LOCAL_DIR` | `./object_store` | Where uploaded bytes are kept when `local` |
| `S3_BUCKET` | `""` | Bucket name (required when `s3`) |
| `S3_REGION` | `us-east-1` | AWS region |
| `S3_ENDPOINT_URL` | `""` | Point at MinIO or another S3-compatible store; blank = AWS |
| `S3_PREFIX` | `documents` | Key prefix for stored documents |
| `EVENT_BUS` | `sqlite` | Event transport: `sqlite` or `kafka` |
| `EVENT_BUS_PATH` | `./checkpoints/event_bus.db` | Queue file when `sqlite` (workers must share it) |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka brokers |
| `KAFKA_TOPIC_INGESTION` | `document.uploaded` | Upload event topic |
| `KAFKA_TOPIC_DLQ` | `document.uploaded.dlq` | Dead-letter topic |
| `KAFKA_CONSUMER_GROUP` | `ingestion-workers` | Consumer group for workers |
| `INGEST_MAX_ATTEMPTS` | `3` | Attempts before an event is dead-lettered |
| `INGEST_RETRY_BACKOFF_S` | `2.0` | Base delay for exponential retry backoff |
| `INGEST_VISIBILITY_TIMEOUT_S` | `300` | Redelivery window if a worker dies mid-document |
| `WORKER_POLL_INTERVAL_S` | `1.0` | Idle poll interval |
| `WORKER_BATCH_SIZE` | `1` | Events claimed per poll |
| `DOCUMENT_REGISTRY_PATH` | `./checkpoints/documents.db` | Document lifecycle store |
| `JSON_LOGS` | `false` | Structured JSON logs with request IDs |

---

## Running the Application

### Option 1: Run both services

```bash
# Terminal 1 - API Server
uvicorn api.app:app --host 0.0.0.0 --port 8000

# Terminal 2 - Streamlit UI
streamlit run ui/app.py --server.port 8501
```

Then open **http://localhost:8501** in your browser.

### Option 2: API only

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Option 3: With the async ingestion worker

Uploads return `202` immediately and a worker indexes them in the background.
No broker required — the defaults use a local object store and a durable
SQLite queue:

```bash
# Terminal 1 - API Server
ASYNC_INGESTION=true uvicorn api.app:app --host 0.0.0.0 --port 8000

# Terminal 2 - Ingestion worker (run more copies to index faster)
ASYNC_INGESTION=true python -m scripts.worker

# Terminal 3 - Streamlit UI
streamlit run ui/app.py --server.port 8501
```

### Option 4: CLI

```bash
# Naive mode (fast, no graph)
python -m scripts.ask "What is the remote work policy?"

# Graph mode (CRAG pipeline)
python -m scripts.ask --mode graph "Compare the onboarding process with the probation policy"

# Auto mode (routes simple queries to naive, complex to graph)
python -m scripts.ask --mode auto "What is the annual leave entitlement?"

# With metadata filter
python -m scripts.ask --mode graph --filter department=hr "What is the dress code?"
```

---

## Docker Deployment

### Development (recommended for local work)

```bash
# Build and start API + UI
docker-compose up --build -d

# Add the ingestion worker (requires ASYNC_INGESTION=true in .env)
docker-compose --profile async up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Services:
- **API**: http://localhost:8000
- **UI**: http://localhost:8501

### Production (Kafka + MinIO + scalable workers)

`docker-compose.prod.yml` brings up the full event-driven topology already
configured — Kafka in KRaft mode (no ZooKeeper), MinIO with its bucket
created on startup, the API, a worker, and the UI:

```bash
docker-compose -f docker-compose.prod.yml up --build -d

# Scale ingestion independently of query traffic
docker-compose -f docker-compose.prod.yml up -d --scale worker=3

# Follow worker logs
docker-compose -f docker-compose.prod.yml logs -f worker
```

Workers get `stop_grace_period: 60s`, so `docker stop` lets the in-flight
document finish rather than tearing an index write in half.

Requires the optional extras in `requirements.txt`: `boto3`, `kafka-python`.

### Using Dockerfile (API only)

```bash
docker build -t rag-assistant .
docker run -p 8000:8000 --env-file .env \
  -v ./chroma_db:/app/chroma_db \
  -v ./checkpoints:/app/checkpoints \
  rag-assistant
```

### Persistent Data

The following directories should be mounted as volumes for data persistence:

| Volume | Purpose |
|--------|---------|
| `./chroma_db` | Vector store data |
| `./checkpoints` | Conversation history, metrics, knowledge graph, document registry, SQLite queue |
| `./object_store` | Uploaded document bytes (`STORAGE_BACKEND=local`) |
| `./data` | Source documents (read-only) |

In production the last two are replaced by MinIO/S3 and Kafka, which manage
their own volumes (`minio_data`, `kafka_data`).

---

## API Reference

Base URL: `http://localhost:8000`

### Health Check

```
GET /health
GET /health?deep=true
```

**Response** (200):
```json
{
  "status": "healthy",
  "collection": "enterprise_docs",
  "document_count": 15,
  "version": "1.0.0"
}
```

Deep health check additionally verifies ChromaDB, SQLite, and LLM connectivity.

---

### Ask a Question

```
POST /ask
Authorization: Bearer <api-key>    # if AUTH_ENABLED=true
Content-Type: application/json
```

**Request Body**:
```json
{
  "question": "What is the company's remote work policy?",
  "mode": "graph",
  "retriever_strategy": "hybrid",
  "filter": {"department": "hr"},
  "top_k": 4,
  "stream": false,
  "session_id": "abc-123"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `question` | string | required | The question to ask |
| `mode` | string | `"naive"` | `"naive"`, `"graph"`, or `"auto"` |
| `retriever_strategy` | string | `"dense"` | One of the 8 retrieval strategies |
| `filter` | object | `null` | Metadata filter (e.g., `{"department": "hr"}`) |
| `top_k` | integer | `null` | Override default top-k |
| `stream` | boolean | `false` | Enable SSE streaming |
| `session_id` | string | `null` | Session ID for multi-turn conversations |

**Response** (200):
```json
{
  "answer": "The company allows remote work up to 3 days per week...",
  "question": "What is the company's remote work policy?",
  "mode": "graph",
  "retriever_strategy": "hybrid",
  "cost_usd": 0.00012,
  "latency_ms": 2340,
  "tokens_used": 580,
  "node_latencies": {
    "scope_check": 0.002,
    "intent_detect": 0.45,
    "retrieve": 0.32,
    "grade_documents": 0.89,
    "generate": 1.2
  },
  "session_id": "abc-123",
  "intent": "informational",
  "cache_hit": false,
  "is_idk": false
}
```

**SSE Streaming** (`stream: true`):

```
data: {"type": "node", "node": "retrieve", "data": {"strategy": "hybrid"}}
data: {"type": "node", "node": "generate", "data": {}}
data: {"type": "token", "content": "The"}
data: {"type": "token", "content": " company"}
data: {"type": "done", "answer": "The company...", "meta": {...}}
```

---

### Ingest Documents

```
POST /ingest
Authorization: Bearer <api-key>
Content-Type: application/json
```

**Request Body**:
```json
{
  "path": "./data/sample_docs",
  "chunk_size": 1000,
  "chunk_overlap": 200
}
```

**Response** (200):
```json
{
  "documents_loaded": 11,
  "chunks_created": 15,
  "chunks_added": 15,
  "collection_total": 15
}
```

---

### Upload Document

```
POST /upload
Authorization: Bearer <api-key>
Content-Type: multipart/form-data
```

Upload a PDF, DOCX, CSV, TXT or MD file (max 10 MB). `department` tags the
document for metadata filtering and access level, and is accepted as either
a form field or a query parameter.

**Response** (200) — default, synchronous indexing:
```json
{
  "filename": "travel_policy.pdf",
  "documents_loaded": 1,
  "chunks_created": 3,
  "chunks_added": 3,
  "collection_total": 18
}
```

**Response** (202) — with `ASYNC_INGESTION=true`, the file is stored and
queued and a worker indexes it (see [Async Ingestion](#async-ingestion)):
```json
{
  "document_id": "doc_7c7cf309aca80ed93c7e46bc10fd2ac5",
  "filename": "travel_policy.pdf",
  "department": "hr",
  "status": "PENDING",
  "duplicate": false,
  "checksum": "e3b0c442...",
  "size_bytes": 20480,
  "status_url": "/documents/doc_7c7cf309aca80ed93c7e46bc10fd2ac5"
}
```

Re-uploading identical content returns **200** with `duplicate: true` and the
original `document_id` — no parsing or embedding is repeated.

---

### Document Status

```
GET /documents/{document_id}
GET /documents?status=PROCESSED&department=hr&limit=50
Authorization: Bearer <api-key>
```

Lifecycle state of uploaded documents. Status is one of `PENDING`,
`PROCESSING`, `PROCESSED`, `FAILED`, `DEAD_LETTER`.

**Response** (200):
```json
{
  "document_id": "doc_7c7cf309aca80ed93c7e46bc10fd2ac5",
  "filename": "travel_policy.pdf",
  "department": "hr",
  "status": "PROCESSED",
  "attempts": 1,
  "chunks_indexed": 3,
  "size_bytes": 20480
}
```

---

### List Tools

```
GET /tools
Authorization: Bearer <api-key>
```

Returns available tools from the MCP registry.

**Response** (200):
```json
[
  {
    "name": "calculator",
    "description": "Evaluate arithmetic expressions",
    "parameters": {"expression": "string"}
  },
  {
    "name": "data_lookup",
    "description": "Look up department-specific documents",
    "parameters": {"department": "string", "query": "string"}
  }
]
```

---

### Run Evaluation

```
POST /eval
Authorization: Bearer <api-key>
Content-Type: application/json
```

**Request Body**:
```json
{
  "mode": "graph",
  "retriever_strategy": "hybrid",
  "limit": 10
}
```

**Response** (200):
```json
{
  "scores": {
    "faithfulness": 0.87,
    "answer_relevancy": 0.91,
    "context_precision": 0.83
  },
  "items_evaluated": 10,
  "mode": "graph",
  "retriever_strategy": "hybrid",
  "duration_s": 45.2
}
```

---

## CLI Tools

| Command | Description |
|---------|-------------|
| `python -m scripts.ingest <path>` | Ingest documents from directory or file |
| `python -m scripts.ask "<query>"` | Query the RAG pipeline |
| `python -m scripts.ask --mode graph "<query>"` | Query using the CRAG graph |
| `python -m scripts.ask --mode auto "<query>"` | Auto-route between naive and graph |
| `python -m scripts.ask --filter department=legal "<query>"` | Query with metadata filter |
| `python -m scripts.metrics` | Display cost/latency metrics dashboard |
| `python -m scripts.metrics --last 50` | Show last 50 queries |
| `python -m scripts.cleanup_stale` | Remove documents past TTL |
| `python -m scripts.upload_eval_dataset` | Upload evaluation dataset |
| `python -m scripts.worker` | Run the ingestion worker (async pipeline) |
| `python -m scripts.worker --once` | Drain the queue and exit (CI/backfill) |
| `python -m scripts.migrate_chroma --dry-run` | Preview an embedded → server vector migration |
| `python -m scripts.migrate_chroma` | Copy the embedded collection into the Chroma server |
| `python -m scripts.replay_dlq --list` | Show dead-lettered documents and why they failed |
| `python -m scripts.replay_dlq` | Drain the DLQ and requeue documents for indexing |

---

## Async Ingestion

By default `/upload` parses and embeds inline and returns `200`. That keeps
the request path simple, but it holds the connection for the whole embedding
run, keeps no durable copy of the file, and loses the upload entirely if the
embedding API returns a 429.

Set `ASYNC_INGESTION=true` to switch to the event-driven pipeline:

```
upload ─► validate ─► idempotency check ─► object store ─► queue ─► 202
                                                             │
                                                             ▼
                                            worker ─► parse ─► chunk
                                                   ─► embed ─► vector DB
                                                   ─► mark PROCESSED
                                                        │
                                          failure ─► retry ×3 ─► DLQ
```

> **Required: `CHROMA_MODE=server`.** The worker indexes in its own
> process, and embedded ChromaDB caches its index inside the process that
> opened it — an API using embedded mode will never retrieve a document
> the worker indexed, until it is restarted. The API warns at startup and
> reports `degraded` health if you enable async ingestion without it.

### Running it locally (no broker required)

Storage and queueing default to `local` and `sqlite` — a durable on-disk
queue with visibility timeouts, so retries and dead-lettering behave exactly
as they do on Kafka, with no broker to install. Only ChromaDB needs to be a
real server:

```bash
# terminal 1 — vector store
chroma run --path ./chroma_server --port 8001 --host 127.0.0.1

# terminal 2 — API
ASYNC_INGESTION=true CHROMA_MODE=server CHROMA_PORT=8001 \
  uvicorn api.app:app --reload

# terminal 3 — worker
ASYNC_INGESTION=true CHROMA_MODE=server CHROMA_PORT=8001 \
  python -m scripts.worker
```

`--host` is not optional on Windows. Chroma defaults to `--host localhost`,
which resolves to the IPv6 loopback `::1`, so the server binds IPv6 only.
`CHROMA_HOST` defaults to `localhost` too, so the two agree by coincidence
— until someone sets `CHROMA_HOST=127.0.0.1`, which is IPv4. Nothing is
listening there, and the client reports `Could not connect to a Chroma
server. Are you sure it is running?` while `netstat` plainly shows the port
bound. Pin both to the same address family and the ambiguity is gone.

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@handbook.pdf" -F "department=hr"
# {"document_id":"doc_7c7c...","status":"PENDING", ...}   202

curl http://localhost:8000/documents/doc_7c7c...
# {"status":"PROCESSED","chunks_indexed":12, ...}
```

### Running it on Kafka + S3

`docker-compose.prod.yml` brings up Kafka (KRaft, no ZooKeeper), MinIO, the
API and a worker, already configured:

```bash
docker-compose -f docker-compose.prod.yml up -d
docker-compose -f docker-compose.prod.yml up -d --scale worker=3
```

Requires the optional extras: `pip install boto3 kafka-python`.

### Operating it

| Concern | Where to look |
|---------|---------------|
| Did my upload index? | `GET /documents/{id}` |
| Backlog / stuck documents | `GET /documents` (per-status counts) |
| Queue and DLQ depth | `GET /health?deep=true` |
| What is dead-lettered | `python -m scripts.replay_dlq --list` |
| Tracing one upload across processes | `JSON_LOGS=true` + the `X-Request-ID` header |

### Recovering dead-lettered documents

A non-zero `DEAD_LETTER` count means documents were accepted from users and
never indexed. The bytes are still in object storage, so they can be
recovered without anyone re-uploading:

```bash
python -m scripts.replay_dlq --list      # what is stuck, and why
python -m scripts.replay_dlq --dry-run   # preview; drains nothing
python -m scripts.replay_dlq             # drain the DLQ and requeue everything

python -m scripts.replay_dlq --document-id doc_abc123   # just one
curl -X POST http://localhost:8000/documents/doc_abc123/retry
```

Replay resets the retry budget and publishes a fresh event, so a recovered
document gets the full retry policy again rather than resuming one attempt
from exhaustion. It is safe to run repeatedly: already-indexed documents are
skipped, and a document whose stored object is genuinely gone is reported as
needing a real re-upload rather than being queued to fail again.

A worker must be running, or replayed documents just sit in `PENDING`.

| Setting | Default | Purpose |
|---------|---------|---------|
| `ASYNC_INGESTION` | `false` | Enable the async pipeline |
| `CHROMA_MODE` | `embedded` | **Set to `server` whenever async ingestion is on** |
| `STORAGE_BACKEND` | `local` | `local` or `s3` |
| `EVENT_BUS` | `sqlite` | `sqlite` or `kafka` |
| `INGEST_MAX_ATTEMPTS` | `3` | Attempts before dead-lettering |
| `INGEST_VISIBILITY_TIMEOUT_S` | `300` | Redelivery window if a worker dies |
| `BM25_CACHE_TTL` | `300` | Max age of a sparse index before rebuild |
| `JSON_LOGS` | `false` | Structured logs with request ids |

### Why a ChromaDB server is required

Embedded ChromaDB keeps its index in the memory of the process that opened
it, and chromadb's client is a process-level singleton — so a separate
worker's writes are invisible to the API no matter how the connection is
recycled. Sparse retrieval has the same shape of problem: the BM25 index is
an in-memory copy of the corpus, and its invalidation hook only fires in the
process that did the writing.

Server mode fixes the dense side by making one server authoritative. The
sparse side is handled by checking the corpus size on each lookup (so an
indexed document is picked up immediately) with `BM25_CACHE_TTL` as a
backstop. Both were found by running the app, not by the test suite — see
ARCHITECTURE.md §2.6.

### Migrating an existing corpus to the server

Switching to server mode leaves anything already indexed in `./chroma_db`
behind. Copy it across — stored embeddings are reused, so this costs nothing
in API spend and the vectors are identical:

```bash
chroma run --path ./chroma_server --port 8001 --host 127.0.0.1   # must be running

python -m scripts.migrate_chroma --dry-run          # preview
python -m scripts.migrate_chroma                    # copy
```

Chunk IDs are content hashes, so the migration is idempotent — re-running it,
or running it against a server that already holds some of the same documents,
adds each chunk exactly once and reports how many were already present.

---

## Testing

The project has **1234 tests** across 52 test files (1216 unit + 18 integration) covering unit, integration, and end-to-end scenarios, at **93% line coverage**.

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov=api --cov-report=html

# Run specific phase tests
pytest tests/test_phase10_memory.py -v
pytest tests/test_phase17_security.py -v

# Run only integration tests
pytest tests/test_integration.py -v

# Run smoke tests
pytest tests/test_e2e_smoke.py -v
```

### Test Coverage by Area

| Area | Tests | Files |
|------|-------|-------|
| Retrieval strategies | 53+ | `test_retrieval.py`, `test_phase8_retrieval.py` |
| Graph pipeline | 34+ | `test_graph_nodes.py`, `test_build_graph.py`, `test_phase8_graph_intel.py` |
| Security & auth | 49+ | `test_phase9_security.py`, `test_phase17_security.py`, `test_critical_fixes.py` |
| Memory & sessions | 22+ | `test_phase10_memory.py` |
| Intent & query transform | 50+ | `test_phase11_intent.py`, `test_phase12_query_transformer.py` |
| Cross-encoder reranking | 18+ | `test_phase13_cross_encoder.py` |
| Context builder | 19+ | `test_phase14_context_builder.py` |
| Semantic cache | 20+ | `test_phase15_cache_activation.py` |
| Knowledge graph | 25+ | `test_phase16_knowledge_graph.py` |
| MCP integration | 17+ | `test_phase18_mcp.py` |
| Resilience | 64+ | `test_phase8_resilience.py`, `test_phase9_circuit_breaker.py` |
| Parallel processing | 20+ | `test_phase20_parallel.py` |
| Object storage | 21+ | `test_phase22_storage.py` |
| Event bus & queue semantics | 26+ | `test_phase22_events.py` |
| Document registry & idempotency | 37+ | `test_phase22_registry.py` |
| Ingestion worker (retry/DLQ) | 24+ | `test_phase22_worker.py` |
| Async upload API | 39+ | `test_phase22_upload_async.py` |
| Object storage (local + S3) | 40+ | `test_phase22_storage.py` |
| Kafka transport | 49+ | `test_phase23_kafka_bus.py` |
| Evaluation harness | 31+ | `test_phase23_eval_harness.py` |
| Health checks | 33+ | `test_phase23_health_checks.py` |
| Two-stage retrieval | 28+ | `test_phase23_retrievers.py` |
| Vector store wrapper | 36+ | `test_phase23_vectorstore.py` |
| Config, logging & edge paths | 66+ | `test_phase23_config_and_ingestion_edges.py` |
| Integration & E2E | 46+ | `test_integration.py`, `test_e2e_smoke.py` |

The ingestion suites run against the real SQLite queue, a real registry and a
temp-directory object store, so retry, dead-lettering, visibility timeouts and
concurrent-claim races are exercised for real rather than mocked.

Optional dependencies (`kafka-python`, `boto3`, `sentence-transformers`) are
driven through injected fakes rather than skipped. That keeps the Kafka and S3
backends — the ones production actually runs on — covered in a CI environment
where those packages and services are absent.

```bash
# Coverage report
pytest --cov=src --cov=api --cov=config --cov-report=term-missing

# HTML report
pytest --cov=src --cov=api --cov=config --cov-report=html && open htmlcov/index.html
```

### Integration tests (real Kafka / MinIO / ChromaDB)

The default suite fakes the production backends, which proves call contracts
but not that a deployment works. The integration suite runs the same code
against real services and is excluded from the default run:

```bash
docker compose -f docker-compose.test.yml up -d --wait
pytest -m integration
docker compose -f docker-compose.test.yml down -v
```

It covers Kafka round-trips and offset-commit semantics, redelivery of
unacked events, per-document ordering, poison-message handling, S3
round-trips and error mapping, the full upload → Kafka → S3 → worker →
vector store pipeline, and cross-client visibility on a Chroma server.

`docker-compose.test.yml` mirrors the production backing services on offset
ports (Kafka 19092, MinIO 19000, Chroma 18001) so it can run alongside a
local dev stack. CI runs it as a separate job and validates all three
compose files on every push — worth having: the first run found a withdrawn
Kafka image, a Chroma client/server version mismatch, and a healthcheck that
would have deadlocked production startup. See ARCHITECTURE.md §2.8.

---

## Evaluation

The system uses [RAGAS](https://docs.ragas.io/) for automated evaluation with three metrics:

- **Faithfulness**: Is the answer grounded in the retrieved context?
- **Answer Relevancy**: Does the answer address the question?
- **Context Precision**: Are the retrieved documents relevant?

```bash
# Run evaluation via CLI
python -m src.eval.ragas_eval

# Run evaluation via API
curl -X POST http://localhost:8000/eval \
  -H "Content-Type: application/json" \
  -d '{"mode": "graph", "retriever_strategy": "hybrid"}'
```

The evaluation dataset is stored in `src/eval/eval_set.json` with ground-truth question-answer pairs across all departments.

---

## Security

### Authentication

When `AUTH_ENABLED=true`, all endpoints (except `/health`) require a Bearer token:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the leave policy?"}'
```

Set valid keys via `API_KEYS` environment variable (comma-separated).

### Department scoping (retrieval RBAC)

A key can be restricted to the departments it may read from and upload
into:

```bash
API_KEYS=<admin-key>:*,<hr-key>:hr|general,<legal-key>:legal|general
```

A scoped key that asks for another department gets `403`. More
importantly, a scoped key that asks with *no* filter is narrowed to its
own departments rather than searching the whole corpus — an unfiltered
question previously swept every department, so confidential material
could reach an answer without anyone asking for it.

A key with no `:scope` suffix reads every department, which is the
historical behaviour and keeps existing deployments working. The API
logs a warning at startup naming how many keys are unscoped, so a
deployment that believes it is segmented can see that it is not.

### Input Guardrails

The system detects and rejects:
- **Prompt injection attempts**: 11 regex patterns covering common injection techniques
- **PII in queries**: SSN, credit card numbers, phone numbers, email addresses
- **Oversized queries**: Configurable max length (default: 2000 characters)

### Output Filtering

All LLM responses pass through PII redaction before reaching the client. Detected patterns (SSN, credit card, phone) are replaced with `[REDACTED]`.

Email addresses are deliberately **not** redacted in answers: enterprise
answers are routinely "contact facilities@company.com", and removing the
one actionable detail makes the answer useless.

### PII at ingest

Answer-time redaction protects the response and nothing else — the
document text is embedded verbatim, so the vector store and every backup
of it hold the original. `INGEST_PII_MODE` addresses that:

| Mode | Behaviour |
|------|-----------|
| `off` (default) | Unchanged — indexing behaviour is untouched |
| `warn` | Logs what the corpus contains, indexes it anyway |
| `redact` | Strips PII (including email) before embedding |

`redact` is not the default because it changes what is retrievable, and a
corpus indexed before and after would be inconsistent. That is a decision
for whoever owns the data.

### Parsing untrusted files

PDF and DOCX parsers are a long-standing exploit surface, and the worker
runs them on user-supplied bytes. Bounded by `INGEST_PARSE_TIMEOUT_S`
(a pathological file must not stall the queue behind it) and
`INGEST_MAX_EXTRACTED_CHARS` (a small upload that expands into gigabytes
of text). Both are treated as permanent failures and dead-letter
immediately rather than spending the retry budget on the same outcome.

In `docker-compose.prod.yml` the api and worker containers additionally
run read-only, with all capabilities dropped, `no-new-privileges`, and a
`noexec,nosuid` tmpfs for the temp directory the upload is written to.

### Security Best Practices

- `.env` files are gitignored -- secrets never enter version control
- API keys are validated per-request, not cached
- File uploads are validated for type, size, and filename
- Rate limiting prevents abuse (configurable per-minute threshold)
- CORS is locked to specific origins (default: `localhost:8501`)
- Object storage keys are validated against path traversal -- a crafted filename cannot write outside the storage root
- Documents are scoped to the API key that uploaded them; `/documents` returns 404 (not 403) for another tenant's ID, so it cannot be used to probe for existence
- `./object_store` is gitignored -- uploaded user documents never enter version control

---

## Observability

### Metrics

Every query records:
- Total cost (USD)
- Token count (prompt + completion)
- Latency (total + per-node breakdown)
- Retriever strategy used
- Cache hit/miss
- Circuit breaker state

View metrics via CLI:
```bash
python -m scripts.metrics --last 20
```

### Prometheus

`GET /metrics` exposes the same numbers in Prometheus format — cost,
latency percentiles, IDK rate, grader rejection rate, queue and
dead-letter depth, documents by status — plus a `rag_build_info` series
whose labels carry the active prompt-set fingerprint and model names, so
a change in quality can be attributed to the deploy that caused it.

Authenticated for the same reason `?deep=true` is: the body carries queue
depths and per-department document counts.

```bash
curl -H "Authorization: Bearer $RAG_API_KEY" http://localhost:8000/metrics
```

Scrape config and alert rules are in `deploy/prometheus/`. The alerts
that matter most here are RAG-specific: a non-zero dead-letter queue
(documents accepted and never indexed) and an IDK-rate spike (retrieval
broke, not the model).

### Tracing

With `OTEL_ENABLED=true`, the trace context travels *inside the ingestion
event*, so an upload and the indexing that happens minutes later in the
worker process form a single trace. With it off, the OpenTelemetry SDK is
never imported.

### Prompt versioning

Every prompt is registered with a version and a content hash. The
fingerprint of the whole set is recorded on each query metric
(`prompt_version`), which is what makes a quality regression attributable
rather than merely observed. `PROMPT_OVERRIDE_DIR` allows changing
wording without a code change; an override that does not declare the same
template variables as the built-in is rejected — a prompt that quietly
lost `{context}` would still produce fluent, ungrounded answers.

### Quality regression gate

```bash
python -m scripts.ragas_gate --mode graph --retriever hybrid
```

Fails on an absolute floor or a drop from the recorded baseline beyond
tolerance. Runs nightly (`.github/workflows/ragas-nightly.yml`), never on
pull requests — it spends real tokens and its scores are
non-deterministic, so gating PRs on it would produce flaky failures that
people learn to re-run until green.

### Health Checks

```bash
# Basic health check
curl http://localhost:8000/health

# Deep health check (verifies all subsystems)
curl http://localhost:8000/health?deep=true
```

With async ingestion enabled, the deep check also reports object-store
reachability, queue backlog and **DLQ depth**. A non-zero dead-letter count
means documents were accepted from users and never indexed — a failure that
is otherwise invisible from the query side.

### Ingestion Monitoring

```bash
# Status of one upload
curl http://localhost:8000/documents/doc_7c7cf309aca80ed93c7e46bc10fd2ac5

# Backlog and per-status counts
curl http://localhost:8000/documents

# Just the ones that gave up
curl "http://localhost:8000/documents?status=DEAD_LETTER"
```

Re-uploading a dead-lettered file resets its retry budget and requeues it.

### Request Tracing

Every response carries an `X-Request-ID` (echoed from the request when you
supply one). With `JSON_LOGS=true`, that ID appears on every log line and is
carried through the queue into the worker, so a single upload can be followed
across processes:

```bash
curl -X POST http://localhost:8000/upload \
  -H "X-Request-ID: trace-abc-123" \
  -F "file=@handbook.pdf" -F "department=hr"

docker-compose logs api worker | grep trace-abc-123
```

### LangSmith Tracing

Enable distributed tracing by setting in `.env`:
```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=ls-your-key-here
LANGSMITH_PROJECT=enterprise-rag-assistant
```

---

## Sample Data

The project includes 11 sample enterprise documents across 6 departments:

| Department | Documents |
|------------|-----------|
| Engineering | API Guidelines, Incident Response Runbook |
| Finance | Procurement Policy, Quarterly Report Q1 2026 |
| HR | Employee Handbook, Onboarding Guide |
| Legal | Data Protection Policy, Vendor Contract Terms |
| Operations | Business Continuity Plan, Change Management Process |
| Security | Information Security Policy |

These documents are designed for realistic enterprise RAG scenarios with cross-departmental references, policy details, and structured information.

---

## Troubleshooting

### Common Issues

**"No OpenAI API key found"**
Ensure `OPENAI_API_KEY` is set in your `.env` file and the file is in the project root.

**Empty responses from graph mode**
Run `python -m scripts.ingest ./data/sample_docs` to populate the vector store before querying.

**401 Unauthorized**
If `AUTH_ENABLED=true`, include the `Authorization: Bearer <key>` header. Verify the key is in your `API_KEYS` list.

**Slow first query**
The first query loads the cross-encoder model (~200MB download). Subsequent queries reuse the cached model.

**ChromaDB lock errors**
Ensure only one process accesses the ChromaDB directory at a time. Stop any running API servers before starting a new one.

**Streamlit connection error**
Verify the API server is running on port 8000 before starting Streamlit. Check `CORS_ORIGINS` includes `http://localhost:8501`.

---

## License

This project is for educational and portfolio purposes. See individual dependency licenses for third-party components.
