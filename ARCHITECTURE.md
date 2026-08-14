# Enterprise RAG Assistant - Architecture Document

## 1. System Architecture Diagram

```
                            ENTERPRISE RAG ASSISTANT
  ============================================================================

  USER INTERFACE                    API LAYER                    EXTERNAL
  +------------------+    +---------------------------+    +----------------+
  | Streamlit UI     |    | FastAPI (uvicorn)         |    | OpenAI API     |
  | - Chat interface |--->| - POST /ask               |--->| gpt-4o-mini    |
  | - File upload    |    | - POST /ingest            |    | text-embedding |
  | - SSE streaming  |    | - POST /upload            |    | -3-small       |
  | - Session mgmt   |    | - POST /eval              |    +----------------+
  +------------------+    | - GET  /health            |    | Tavily API     |
                          | - GET  /tools             |    | (web search)   |
                          +---------------------------+    +----------------+
                                     |
                    +----------------+------------------+
                    |                                   |
              SECURITY LAYER                    RATE LIMITING
  +----------------------------------+    +--------------------+
  | API Key Auth (Bearer token)      |    | slowapi            |
  | Input Guardrails                 |    | 30/min default     |
  |   - Prompt injection (11 regex)  |    +--------------------+
  |   - PII detection (SSN,CC,phone) |
  |   - Max query length (2000)      |
  | Output PII Filter (3 redactions) |
  +----------------------------------+
                    |
  ============================================================================
                        LANGGRAPH CRAG PIPELINE
  ============================================================================

  +------------------------------------------------------------------------+
  |                                                                        |
  |  START                                                                 |
  |    |                                                                   |
  |    v                                                                   |
  |  [guardrail_check] ---blocked---> generate(rejection) --> END          |
  |    | continue                                                          |
  |    v                                                                   |
  |  [load_memory] ---- SQLite: conversation_history                       |
  |    |                 session_id, role, content, timestamp               |
  |    v                                                                   |
  |  [cache_lookup] ---cache_hit---> save_memory --> END                   |
  |    | cache_miss      Cosine similarity >= 0.95                         |
  |    v                 against cached embeddings                         |
  |  [scope_check] ---out_of_scope---> generate(IDK) --> END               |
  |    | in_scope        44 domain keywords vs                             |
  |    v                 off-topic regex patterns                          |
  |  [intent_detect]                                                       |
  |    | LLM structured output -> IntentResult                             |
  |    | {informational|comparative|procedural|analytical|multi_hop|factual}|
  |    | Heuristic fallback when circuit breaker open                      |
  |    v                                                                   |
  |  [query_transform]                                                     |
  |    | 1. Normalize (acronyms, whitespace)                               |
  |    | 2. Extract entities (LLM + regex fallback)                        |
  |    | 3. Intent-aware rewrite (if comparative/procedural/analytical)     |
  |    | 4. Expand with entity names                                       |
  |    v                                                                   |
  |  [tool_router] ---- MCP registry (when enabled)                        |
  |    | LLM function-calling selects tool                                 |
  |    | Regex fallback: calculator, data_lookup                           |
  |    v                                                                   |
  |  [planner]                                                             |
  |    | LLM decomposes into sub-questions (max 5)                         |
  |    |                                                                   |
  |    +---is_multi_part=true---+---is_multi_part=false---+                |
  |    |                        |                         |                |
  |    v                        v                         |                |
  |  [process_sub_query]  [process_sub_queries_parallel]  |                |
  |    | (sequential loop)  | (ThreadPoolExecutor)        |                |
  |    v                    v                             |                |
  |  [synthesize] <---------+                             |                |
  |    | Combines sub-answers via LLM                     |                |
  |    |                                                  |                |
  |    |              +---------- RETRIEVAL LOOP ---------+                |
  |    |              |                                                    |
  |    |              v                                                    |
  |    |            [retrieve] --- get_retriever(strategy)                 |
  |    |              |           (see Retrieval Pipeline below)            |
  |    |              v                                                    |
  |    |            [grade_documents]                                      |
  |    |              | LLM structured output: GradeResult.relevant        |
  |    |              | Per-doc grading mode (optional, parallel)           |
  |    |              |                                                    |
  |    |              +--relevant=true----> [generate]                      |
  |    |              |                        |                           |
  |    |              +--retries < 2--------> [transform_query]            |
  |    |              |                        | LLM rewrites query        |
  |    |              |                        +---> [retrieve] (loop)     |
  |    |              |                                                    |
  |    |              +--retries >= 2-------> [web_search]                  |
  |    |                                       | Tavily API (3 results)    |
  |    |                                       +---> [generate]            |
  |    |                                                                   |
  |    +------------------+--------------------+                           |
  |                       v                                                |
  |                     [critic]                                           |
  |                       | Claim-level verification                       |
  |                       | LLM extracts claims -> supported/unsupported   |
  |                       | Rewrites answer removing unsupported claims    |
  |                       v                                                |
  |                     [cache_store] --- Save to semantic cache            |
  |                       v                                                |
  |                     [save_memory] --- Persist Q/A to session            |
  |                       v                                                |
  |                      END                                               |
  |                                                                        |
  +------------------------------------------------------------------------+

  ============================================================================
                         RETRIEVAL PIPELINE
  ============================================================================

  8 Retrieval Strategies (factory pattern):

  +------------------+------------------------------------------------+
  | Strategy         | Pipeline                                       |
  +------------------+------------------------------------------------+
  | dense            | Query -> OpenAI Embed -> ChromaDB cosine       |
  | hybrid           | Query -> [Dense + BM25] -> RRF Fusion          |
  | multi_query      | Query -> LLM variants -> Dense -> Deduplicate  |
  | rerank           | Query -> Dense(12) -> LLM score(0-10) parallel |
  | hybrid_rerank    | Query -> Hybrid(12) -> LLM score parallel      |
  | cross_rerank     | Query -> Dense(12) -> CrossEncoder batch score |
  | hybrid_cross_rr  | Query -> Hybrid(12) -> CrossEncoder batch      |
  | knowledge_graph  | Query -> Entity extract -> NetworkX BFS -> Docs|
  +------------------+------------------------------------------------+

  Detailed Hybrid + Cross-Encoder Rerank flow:

  +---------+     +------------------+     +------------------+
  | Query   |---->| OpenAI Embeddings|---->| ChromaDB         |
  | (norml) |     | text-embedding-  |     | cosine_similarity|
  |         |     | 3-small          |     | top 20 results   |
  +---------+     +------------------+     +--------+---------+
       |                                            |
       |          +------------------+              |
       +--------->| BM25Okapi        |              |
                  | regex tokenizer  |              |
                  | stop-word filter |              |
                  | top 20 results   |              |
                  +--------+---------+              |
                           |                        |
                           v                        v
                  +------------------------------------+
                  | RRF Fusion                         |
                  | score = 1/(60+rank_d) + 1/(60+r_b) |
                  | MD5 dedup, sort by score            |
                  | return top 12 candidates            |
                  +----------------+-------------------+
                                   |
                                   v
                  +------------------------------------+
                  | Cross-Encoder Reranker             |
                  | ms-marco-MiniLM-L-6-v2             |
                  | batch_size=16, device=cpu           |
                  | [(query, doc)] pairs -> scores      |
                  | return top 4 by score               |
                  +----------------+-------------------+
                                   |
                                   v
                  +------------------------------------+
                  | Context Builder                    |
                  | SHA-256 dedup -> source grouping    |
                  | proportional token allocation      |
                  | budget: 4000 tokens (char/4 est)   |
                  +------------------------------------+

  ============================================================================
                       INDEXING / INGESTION PIPELINE
  ============================================================================

  Shared indexing steps (used by both the sync and async paths):

  +----------+    +--------------+    +------------------+    +-------------+
  | PDF/DOCX/|--->| Loader       |--->| Chunker          |--->| ChromaDB    |
  | CSV/TXT/ |    | PyPDFLoader  |    | Markdown-aware   |    | add_chunks  |
  | MD files |    | Docx2txt     |    | H1/H2/H3 split  |    |             |
  |          |    | CSVLoader    |    | then Recursive   |    | SHA-256 ID  |
  +----------+    | TextLoader   |    | size=1000        |    | dedup on    |
                  | + metadata:  |    | overlap=200      |    | re-ingest   |
                  |  department  |    | seps: \n\n,\n,.  |    |             |
                  |  access_lvl  |    +------------------+    | + BM25 cache|
                  |  ingested_at |                            |   invalidate|
                  +--------------+                            | + KG extract|
                                                              |   (if on)   |
                                                              +-------------+

  How those steps are driven depends on ASYNC_INGESTION — see §1.2.

  ============================================================================
                     RESILIENCE & OBSERVABILITY
  ============================================================================

  Circuit Breaker (per service):         Metrics Store (SQLite):
  +------------------------------+      +---------------------------+
  | CLOSED -> failures >= 5 ->   |      | query_metrics table       |
  |   OPEN -> 60s timeout ->     |      | - tokens (prompt/compl)   |
  |   HALF_OPEN -> probe ->      |      | - cost_usd                |
  |   CLOSED (on success)        |      | - latency_ms              |
  | Services: llm, retrieval,    |      | - is_idk, grader_rejected |
  |   tavily                     |      | - per-node latencies      |
  +------------------------------+      +---------------------------+

  Node Tracing (@traced):               Health Checker:
  +------------------------------+      +---------------------------+
  | Per-node latency histograms  |      | /health?deep=true         |
  | p50, p95, p99, mean, last    |      | - ChromaDB connectivity   |
  | Document counts in/out       |      | - SQLite accessibility    |
  | Generation length            |      | - Memory usage (psutil)   |
  +------------------------------+      +---------------------------+

  ============================================================================
                         DATA STORES
  ============================================================================

  +-------------------+  +-------------------+  +-------------------+
  | ChromaDB          |  | SQLite            |  | NetworkX Graph    |
  | ./chroma_db/      |  | ./checkpoints/    |  | ./checkpoints/    |
  |                   |  |                   |  | knowledge_graph   |
  | Vector store      |  | graph_checkpoints |  | .json             |
  | 1536-dim vectors  |  |   .db             |  |                   |
  | Content-hash IDs  |  | - LangGraph state |  | Directed graph    |
  | Metadata filter   |  | - Query metrics   |  | Entity nodes      |
  | Auto-refresh 300s |  |                   |  | Relation edges    |
  +-------------------+  | conversations.db  |  | BFS traversal     |
                          | - Chat history    |  | JSON persistence  |
                          |                   |  +-------------------+
                          | semantic_cache.db |
                          | - Cached Q/A      |
                          | - Embedding vecs  |
                          +-------------------+
```

---

## 1.2 Asynchronous Ingestion Pipeline

Enabled with `ASYNC_INGESTION=true`. The upload request stops doing the
expensive work: it persists the file, records it, publishes an event and
returns `202`. A separate worker process does the parsing and embedding.

```
  CLIENT                       API (FastAPI)
  +--------+  POST /upload   +--------------------------------------+
  | Client |---------------->| 1. Validate (.pdf/.docx/.csv/        |
  +--------+                 |    .txt/.md, MIME, size, department) |
      ^                      | 2. SHA-256 -> document_id            |
      |  202 Accepted        | 3. Idempotency check (registry)      |
      |  {document_id}       | 4. Store bytes  (S3 / local)         |
      +----------------------| 5. Publish event (Kafka / SQLite)    |
                             +------------------+-------------------+
                                                |
                    Document Already Exists?    |
                    +-----------+---------------+
                    | YES                       | NO
                    v                           v
        +-------------------------+   +-----------------------+
        | 200 + existing          |   | Store to object store |
        | document_id             |   | Publish event         |
        | duplicate=true          |   | 202 Accepted          |
        | (no work re-queued)     |   +-----------+-----------+
        +-------------------------+               |
  ..............................................  |  .....................
                                                  v
                            +---------------------------------+
                            |  Topic: document.uploaded       |
                            |  (Kafka, or durable SQLite      |
                            |   queue with visibility timeout)|
                            +----------------+----------------+
                                             |
                            +----------------v----------------+
                            |      INGESTION WORKER           |
                            |   python -m scripts.worker      |
                            +----------------+----------------+
                                             |
                                   Already Indexed?
                             +---------------+---------------+
                             | YES                           | NO
                             v                               v
                    +-----------------+       +-------------------------+
                    | Ack / ignore    |       | Download from storage   |
                    | (no re-embed)   |       | Parse document          |
                    +-----------------+       | Split into chunks       |
                                              | Generate embeddings     |
                                              | Store in vector DB      |
                                              | Mark PROCESSED          |
                                              +-----------+-------------+
                                                          |
                                              Any failure during processing?
                                                          |
                    +-------------------------------------+
                    |
                    v
          +--------------------+   attempts < 3    +---------------------+
          | Permanent?         |------------------>| Retry with          |
          | (parse error /     |   (transient)     | exponential backoff |
          |  object missing)   |                   | attempt+1 re-queued |
          +---------+----------+                   +---------------------+
                    | yes, or attempts >= 3
                    v
          +-------------------------------+
          | Dead Letter Queue             |
          | topic: document.uploaded.dlq  |
          | registry status: DEAD_LETTER  |
          +-------------------------------+
```

### Components

| Component | Module | Backends |
|-----------|--------|----------|
| Object storage | `src/storage/object_store.py` | `local` (filesystem, default), `s3` (boto3 / MinIO) |
| Event bus | `src/events/` | `sqlite` (durable queue, default), `kafka` |
| Document registry | `src/ingestion/registry.py` | SQLite (WAL, shared across processes) |
| Indexing steps | `src/ingestion/pipeline.py` | reuses loader/chunker/chroma_store unchanged |
| Worker | `src/ingestion/worker.py`, `scripts/worker.py` | — |

Both backend pairs exist for the same reason: the default (`local` +
`sqlite`) needs no broker and no cloud account, so a developer clone and
the CI suite exercise the *same* retry and dead-letter code paths that
production runs on Kafka and S3.

### Idempotency

Identity is `sha256(file_bytes)` + department, so the document id is
derived rather than random. Two things follow:

- **At the API.** A repeat upload is recognised before anything is stored
  or queued, and returns `200` with the original id. The parse and embed
  cost is never paid twice. (Chunk-level dedup in the vector store still
  applies underneath, but it only fires *after* embedding.)
- **At the worker.** Queues deliver at-least-once, so a redelivered event
  is expected. `claim_for_processing` is a conditional `UPDATE`, so only
  one worker wins; a document already `PROCESSED` is acked, not re-embedded.

### Failure handling

| Failure | Behaviour |
|---------|-----------|
| Transient (embedding 429, vector store blip) | Retried up to `INGEST_MAX_ATTEMPTS` (3) with exponential backoff |
| Permanent (corrupt file, object missing) | Dead-lettered immediately — retrying cannot change the outcome |
| Worker killed mid-document | Visibility timeout redelivers the event; the stale `PROCESSING` claim is reclaimable so the document is not stranded |
| DLQ publish fails | Registry still records `DEAD_LETTER`, so the document is never silently lost |
| Retry publish fails | Original event is left unacked for redelivery rather than dropped |

### Observability

`GET /documents/{id}` returns the lifecycle state; `GET /documents`
lists documents with per-status counts. `GET /health?deep=true` adds
object-store reachability, queue backlog and **DLQ depth** — a non-zero
dead-letter count means documents were accepted from users and never
indexed, which is otherwise invisible from the query side. With
`JSON_LOGS=true` each line carries the `X-Request-ID`, so one upload can
be followed across API, queue and worker.

---

## 2. Architecture Evaluation

### 2.1 Strengths

| Area | Implementation | Assessment |
|------|---------------|------------|
| **Retrieval Diversity** | 8 strategies with factory pattern | Excellent. Covers dense, sparse, hybrid, reranked, and graph-based retrieval |
| **Corrective RAG** | Grade -> rewrite -> retry -> web fallback | Strong self-healing loop with progressive fallback |
| **Claim Verification** | Critic node with per-claim extraction | Reduces hallucination significantly |
| **Resilience** | Circuit breakers on all external calls | Prevents cascading failures |
| **Feature Flags** | Every feature gated, safe defaults | Zero-risk incremental rollout |
| **Thread Safety** | Locks on all singletons and shared state | No race conditions |
| **Security Layers** | Auth + guardrails + output filter | Defense in depth |
| **Multi-turn Memory** | SQLite-backed per-session history | Enables follow-up conversations |
| **Observability** | Per-node tracing + cost tracking + health checks | Full pipeline visibility |

### 2.2 Current Limitations vs Production Best Practices

#### A. Semantic Cache - O(n) Linear Scan
**Current**: Loads ALL cache entries, computes cosine similarity against each one.
**Problem**: At 10K+ cached queries, every cache lookup becomes slow.
**Best Practice**: Use a vector index (FAISS, Milvus, or a dedicated ChromaDB collection) for sub-millisecond ANN lookup.

```
Current:  Query -> embed -> for each in cache: cosine(q, c) -> best match
Optimal:  Query -> embed -> FAISS.search(q, k=1) -> threshold check
```

#### B. Token Estimation - Character Heuristic
**Current**: `tokens = len(text) // 4` (rough character-to-token ratio).
**Problem**: Can be 15-30% off for code, non-English text, or technical content. Over-allocating wastes context; under-allocating truncates.
**Best Practice**: Use `tiktoken` with the actual model's tokenizer (`cl100k_base` for gpt-4o-mini).

#### C. Conversation Memory - No TTL / Unbounded Growth
**Current**: Messages accumulate indefinitely per session. No cleanup.
**Problem**: SQLite DB grows without bound. Old sessions never pruned.
**Best Practice**: Add scheduled cleanup (e.g., delete sessions older than 30 days) or enforce max total rows.

#### D. Knowledge Graph - No Timeout on BFS
**Current**: BFS traversal up to depth 2, but no wall-clock timeout.
**Problem**: Dense subgraphs could expand to thousands of nodes, hanging the request.
**Best Practice**: Add a node-count limit (e.g., max 100 visited nodes) alongside depth limit.

#### E. Single LLM Provider
**Current**: Hardcoded to OpenAI (gpt-4o-mini + text-embedding-3-small).
**Problem**: Single point of failure. No cost optimization across providers. No local model fallback.
**Best Practice**: Abstract LLM behind a provider interface. Support Azure OpenAI, Anthropic, or local models (Ollama) as fallbacks.

#### F. No Async Pipeline
**Current**: Graph nodes run synchronously. API uses `asyncio.to_thread()` to avoid blocking.
**Problem**: Thread pool exhaustion under high concurrency. Each request occupies a thread for the full pipeline duration (5-15s).
**Best Practice**: Native async nodes with `aiohttp` for LLM calls. LangGraph supports async node functions.

#### G. Intent Detection Not Used for Routing
**Current**: Intent is classified and stored in state but never read downstream.
**Problem**: Dead feature. No adaptive retrieval strategy or prompt selection based on intent.
**Best Practice**: Use intent to select retrieval strategy (multi_hop -> hybrid, factual -> dense), prompt template (procedural -> step-by-step), and generation parameters.

#### H. No Document-Level Access Control
**Current**: Department metadata exists but `filter` field was only recently wired. No user-to-department RBAC.
**Problem**: Any authenticated user can query any department's documents.
**Best Practice**: Map API keys/JWT claims to allowed departments. Enforce at retrieval time.

---

### 2.3 Hardening Applied (2026-07)

The following production gaps were identified in review and fixed:

| Fix | Where |
|-----|-------|
| `/ingest` restricted to `INGEST_ROOT` (was an arbitrary-file-read primitive) | `api/app.py`, `config.py` |
| Sessions bound to API-key identity (was: any caller could read any session by guessing its ID) | `api/app.py`, `src/security/auth.py` |
| Semantic cache scoped by retrieval filter; skipped for conversation-context turns; keyed on original question | `src/cache/semantic_cache.py`, `src/graph/cache_nodes.py` |
| SSE streaming is genuinely incremental (was: full pipeline ran, then a burst replay) and all streamed content passes the PII filter | `api/app.py` |
| Per-request node latencies are context-scoped (was: cross-contaminated under concurrency) and the global accumulator is bounded (was: memory leak) | `src/graph/tracing.py` |
| Cost tracking propagates into parallel sub-query / per-doc grading threads via `contextvars.copy_context` | `src/graph/planner.py`, `src/graph/nodes.py` |
| Grader errors no longer trigger the rewrite→retry→web-search path (critic still verifies) | `src/graph/nodes.py` |
| Rate limits on `/ingest`, `/upload`, `/eval` (`HEAVY_RATE_LIMIT`) | `api/app.py` |
| Constant-time API key comparison | `src/security/auth.py` |
| Default checkpointer is in-memory (`GRAPH_CHECKPOINTER=sqlite` to opt back in) — request-scoped runs never resume threads | `src/graph/build_graph.py` |
| Prod runs 1 uvicorn worker with `--proxy-headers` (process-local rate limits/breakers/caches diverge across workers); UI container no longer receives the full `.env` | `Dockerfile.prod`, `docker-compose.prod.yml` |
| `top_k` honored in graph mode; unknown LLM models warn instead of silently costing $0 | `src/graph/nodes.py`, `src/observability/cost_callback.py` |

---

### 2.4 Latency / Cost Optimizations (opt-in)

Per-query the pipeline made 6-8 serial LLM calls. Two opt-in flags cut that
down; both default off so behavior is unchanged unless enabled.

| Flag | Effect | Where |
|------|--------|-------|
| `UNIFIED_ANALYSIS=true` | One structured LLM call replaces `intent_detect` → `query_transform` → `planner` (intent + entities + query rewrite + decomposition in a single pass), removing 2 serial round-trips. Falls back to the same zero-LLM heuristics on error/circuit-open. | `src/graph/analyzer.py`, `src/graph/build_graph.py` |
| `CRITIC_MODE=adaptive` | Skips claim-verification (1-2 LLM calls) for low-risk answers — grader passed, no web fallback, not multi-part, and shorter than `CRITIC_SKIP_MAX_CHARS`. `always` (default) verifies everything; `off` disables the critic. | `src/graph/nodes.py`, `config.py` |
| `retriever_strategy="auto"` | `resolve_strategy` routes by intent: `factual`→dense, `comparative`/`analytical`→hybrid+rerank, else hybrid. | `src/retrieval/factory.py` |

---

### 2.5 Async Ingestion Pipeline (Phase 22)

Ingestion was the last part of the system still doing unbounded work
inside a request. `POST /upload` read the file into memory, wrote it to a
`TemporaryDirectory`, parsed, embedded and indexed it, then deleted the
file. The consequences were all on the ingestion side:

| Problem | Fix |
|---------|-----|
| Request blocked for the full embed duration (UI timed out at 60s on large PDFs) | `202 Accepted` — indexing moved to a worker |
| No durable copy of the upload; the only copy was deleted with the temp dir | Bytes persisted to object storage before the response |
| A transient OpenAI 429 lost the upload with no record it had happened | Retry with backoff, then DLQ; every attempt recorded |
| Idempotency only at chunk level, so re-uploads paid the full parse + embed cost | Content-addressed document ids — duplicates rejected before any work |
| No way to ask "did my upload succeed?" | `GET /documents/{id}` lifecycle status |
| Ingestion competed with query traffic in one process | Workers scale independently (`--scale worker=N`) |
| `department` read as a query param only, so the form field the UI sends was discarded — every upload filed as "general", and legal/security docs never got `confidential` access level | Bound from both form and query |
| Sync upload ran blocking I/O directly in an `async def`, stalling the event loop for every concurrent request | Moved to `asyncio.to_thread` |

The synchronous path is unchanged and remains the default
(`ASYNC_INGESTION=false`); `docker-compose.prod.yml` enables the async
pipeline with Kafka and MinIO.

---

## 3. Production Gap Analysis

### Critical Gaps

| Gap | Impact | Effort | Status |
|-----|--------|--------|--------|
| No horizontal scaling (single-process) | Cannot handle >50 concurrent users | High | Partly closed — ingestion workers scale out; query pods still need shared Redis/Postgres |
| SQLite for all stores (not suitable for concurrent writes) | Write contention under load | High | Open (registry/queue use WAL, which is adequate for a single node) |
| No request tracing (X-Request-ID) | Cannot debug distributed issues | Low | **Closed** — request-id middleware + `JSON_LOGS` |
| Cost budget not enforced (only logged) | Runaway costs on expensive queries | Medium | Open |

### Recommended Gaps (Medium Priority)

| Gap | Impact | Effort | Status |
|-----|--------|--------|--------|
| No retry with exponential backoff on LLM calls | Transient failures not recovered | Low | Closed for ingestion; query path still relies on the circuit breaker |
| No structured logging (JSON) | Hard to parse in log aggregators | Low | **Closed** — `JSON_LOGS=true` |
| No Prometheus metrics export | No dashboarding or alerting | Medium | Open (`/health?deep=true` exposes queue/DLQ depth in the meantime) |
| No document versioning | Can't track what changed when | Medium | Partly closed — every document has a checksum, storage key and audit timestamps |
| No A/B testing framework | Can't compare retrieval strategies in production | Medium | Open |

---

## 4. Optimized Architecture Recommendation

### 4.1 Immediate Improvements (No Architecture Change)

```python
# 1. Wire intent-based routing (use existing data)
def retrieve(state):
    intent = state.get("intent", "informational")
    if intent == "multi_hop":
        strategy = "hybrid"           # broader recall
    elif intent == "factual":
        strategy = "dense"            # precise match
    elif intent == "comparative":
        strategy = "hybrid_cross_rerank"  # best ranking
    else:
        strategy = state.get("retriever_strategy", "hybrid")

# 2. Replace cache linear scan with ChromaDB collection
class SemanticCache:
    def __init__(self):
        self._collection = chroma_client.get_or_create("cache")

    def lookup(self, query_embedding):
        results = self._collection.query(query_embedding, n_results=1)
        if results["distances"][0][0] < (1 - threshold):
            return results["documents"][0][0]

# 3. Use tiktoken for accurate token counting
import tiktoken
_enc = tiktoken.encoding_for_model("gpt-4o-mini")
def count_tokens(text: str) -> int:
    return len(_enc.encode(text))
```

### 4.2 Scale-Ready Architecture (Next Evolution)

```
                    Load Balancer (nginx / cloud LB)
                           |
              +------------+------------+
              |            |            |
         API Pod 1    API Pod 2    API Pod 3
         (FastAPI)    (FastAPI)    (FastAPI)
              |            |            |
              +-----+------+------+-----+
                    |             |
            +-------+-------+  +-+----------+
            | PostgreSQL    |  | Redis       |
            | - Metrics     |  | - Cache     |
            | - Sessions    |  | - Rate limit|
            | - Chat history|  | - Pub/sub   |
            +---------------+  +-------------+
                    |
            +-------+-------+
            | Vector Store   |
            | Qdrant / Pgvec |
            | (replicated)   |
            +----------------+
```

**Key changes for scale:**
1. **PostgreSQL** replaces SQLite for metrics, sessions, memory (concurrent writes)
2. **Redis** replaces in-memory rate limiting and semantic cache (shared across pods)
3. **Qdrant/Pgvector** replaces ChromaDB (production-grade, replicated, filtered search)
4. **Stateless API pods** behind load balancer (horizontal scaling)
5. **Async pipeline** with native `aiohttp` LLM calls (no thread pool exhaustion)

### 4.3 Why These Changes Matter

| Change | Current Bottleneck | Improvement |
|--------|-------------------|-------------|
| PostgreSQL | SQLite locks on concurrent writes | 100x write throughput |
| Redis cache | O(n) cache scan per request | O(1) lookup, shared across pods |
| Qdrant | ChromaDB not designed for production scale | Filtered search, replication, snapshots |
| Async nodes | Thread pool = max ~20 concurrent requests | 1000+ concurrent with async I/O |
| Multi-provider LLM | OpenAI outage = total downtime | Automatic failover to Azure/Anthropic |

---

## 5. Component Technology Summary

| Component | Current | Production Alternative |
|-----------|---------|----------------------|
| LLM | OpenAI gpt-4o-mini | + Azure OpenAI / Anthropic failover |
| Embeddings | text-embedding-3-small (1536d) | Same (or Cohere embed-v3 for cost) |
| Vector DB | ChromaDB (local, SQLite backend) | Qdrant / Pgvector / Weaviate |
| Sparse Search | rank_bm25 (in-memory) | Elasticsearch / OpenSearch |
| Reranker | CrossEncoder ms-marco-MiniLM-L-6-v2 | Same (or Cohere rerank-v3) |
| Knowledge Graph | NetworkX (in-memory, JSON persist) | Neo4j / Amazon Neptune |
| Cache | SQLite + cosine scan | Redis + FAISS index |
| Memory | SQLite | PostgreSQL / Redis |
| Metrics | SQLite | PostgreSQL + Prometheus |
| Auth | Static API keys | OAuth2 / JWT with JWKS |
| Orchestration | LangGraph (single process) | LangGraph + Celery workers |
| Deployment | uvicorn single process | Kubernetes + horizontal pod autoscaler |
| Document storage | Local filesystem (`STORAGE_BACKEND=local`) | S3 / MinIO (`STORAGE_BACKEND=s3`) |
| Ingestion queue | SQLite queue (`EVENT_BUS=sqlite`) | Kafka (`EVENT_BUS=kafka`) |
| Ingestion compute | Inline in the request | Dedicated worker processes (`scripts/worker.py`) |

---

## 6. Verdict

The current architecture is **well-designed for its stage** -- a feature-complete prototype with comprehensive retrieval strategies, self-correcting CRAG pipeline, and layered security. The codebase demonstrates strong engineering patterns (factory, singleton, circuit breaker, feature flags).

**For production deployment at scale**, the three highest-ROI changes are:
1. **Replace SQLite with PostgreSQL** (eliminates write contention)
2. **Replace cache linear scan with vector index** (eliminates O(n) per request)
3. **Wire intent-based routing** (uses existing dead feature to improve retrieval quality)

These three changes require minimal architectural disruption while delivering the most impact.
