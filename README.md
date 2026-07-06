# Enterprise RAG Assistant

A production-grade **Retrieval-Augmented Generation (RAG)** system built for enterprise document intelligence. Features a Corrective RAG (CRAG) pipeline with LangGraph orchestration, 8 retrieval strategies, multi-agent question decomposition, conversation memory, knowledge graph, and a full security layer.

**Tech Stack**: LangChain 1.0 | LangGraph 1.0 | ChromaDB | OpenAI | FastAPI | Streamlit

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
- **Health checks**: Deep subsystem checks (ChromaDB, SQLite, LLM availability)
- **LangSmith integration**: Optional distributed tracing

### API & UI
- **FastAPI REST API**: 6 endpoints with SSE streaming, CORS, rate limiting
- **Streamlit chat UI**: Dark theme, session persistence, pipeline progress visualization, file upload
- **Docker support**: Multi-service compose with health checks

---

## Architecture Overview

```
+------------------+     +---------------------------+     +----------------+
| Streamlit UI     |     | FastAPI API Layer          |     | OpenAI API     |
| - Chat interface |---->| - POST /ask (SSE stream)   |---->| gpt-4o-mini    |
| - File upload    |     | - POST /ingest             |     | text-embedding |
| - Session mgmt   |     | - POST /upload             |     | -3-small       |
+------------------+     | - POST /eval               |     +----------------+
                          | - GET  /health             |     | Tavily API     |
                          | - GET  /tools              |     | (web fallback) |
                          +---------------------------+     +----------------+
                                     |
                    +----------------+------------------+
                    |                                   |
              Security Layer                    Rate Limiting
         (Auth + Guardrails + PII)             (30 req/min)
                    |
                    v
          LangGraph CRAG Pipeline
          (see Pipeline Flow below)
                    |
          +---------+---------+-----------+
          |         |         |           |
       ChromaDB  SQLite    NetworkX   BM25Okapi
       (vectors) (memory,  (knowledge  (sparse
                  metrics,  graph)     retrieval)
                  cache)
```

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
│   │   ├── loader.py          # PDF/TXT/MD file loaders with metadata
│   │   └── chunker.py         # Recursive + markdown-aware splitting
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
│   │   ├── health_checker.py  # Subsystem health checks
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
│   ├── cleanup_stale.py       # Document TTL cleanup
│   ├── metrics.py             # Metrics dashboard CLI
│   └── upload_eval_dataset.py # Evaluation dataset uploader
│
├── tests/                     # 700+ tests across 36 files
│   ├── conftest.py            # Shared fixtures
│   └── test_*.py              # Unit, integration, and e2e tests
│
├── data/sample_docs/          # Sample enterprise documents (6 departments)
├── Dockerfile                 # Python 3.11-slim container
├── docker-compose.yml         # API + UI multi-service deployment
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

### Option 3: CLI

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

### Using Docker Compose (recommended)

```bash
# Build and start both services
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Services:
- **API**: http://localhost:8000
- **UI**: http://localhost:8501

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
| `./checkpoints` | Conversation history, metrics, knowledge graph |
| `./data` | Source documents (read-only) |

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
POST /upload?department=hr
Authorization: Bearer <api-key>
Content-Type: multipart/form-data
```

Upload a PDF, TXT, or MD file (max 10 MB). The `department` query parameter tags the document for metadata filtering.

**Response** (200):
```json
{
  "filename": "travel_policy.pdf",
  "documents_loaded": 1,
  "chunks_created": 3,
  "chunks_added": 3,
  "collection_total": 18
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

---

## Testing

The project has **700+ tests** across 36 test files covering unit, integration, and end-to-end scenarios.

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
| Integration & E2E | 46+ | `test_integration.py`, `test_e2e_smoke.py` |

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

### Input Guardrails

The system detects and rejects:
- **Prompt injection attempts**: 11 regex patterns covering common injection techniques
- **PII in queries**: SSN, credit card numbers, phone numbers, email addresses
- **Oversized queries**: Configurable max length (default: 2000 characters)

### Output Filtering

All LLM responses pass through PII redaction before reaching the client. Detected patterns (SSN, credit card, phone) are replaced with `[REDACTED]`.

### Security Best Practices

- `.env` files are gitignored -- secrets never enter version control
- API keys are validated per-request, not cached
- File uploads are validated for type, size, and filename
- Rate limiting prevents abuse (configurable per-minute threshold)
- CORS is locked to specific origins (default: `localhost:8501`)

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

### Health Checks

```bash
# Basic health check
curl http://localhost:8000/health

# Deep health check (verifies all subsystems)
curl http://localhost:8000/health?deep=true
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
