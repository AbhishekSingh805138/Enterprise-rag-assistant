# The Enterprise RAG Assistant — A Complete Study Guide

> A learning companion for understanding this codebase and how to build a
> production-grade RAG system from scratch. Every concept maps to a real file
> so you can read the code alongside the theory. Work through it top to bottom.

---

## Part 0 — The mental model (read this first)

**RAG = Retrieval-Augmented Generation.** An LLM is frozen knowledge with a
training cutoff and no access to *your* private data. RAG fixes that by doing
two things at query time:

1. **Retrieve** the most relevant chunks from *your* documents.
2. **Augment** the LLM's prompt with those chunks, so it **generates** an answer
   grounded in *your* data — with citations.

The entire project is an elaboration of that one sentence. Everything else
(hybrid search, reranking, critic, caching, guardrails) exists to make
retrieval *more accurate* and generation *more trustworthy and production-safe*.

Two big architectural stages:

- **Ingestion (offline):** documents → chunks → embeddings → vector DB. Done
  once, ahead of time.
- **Query (online):** question → retrieve → (correct/verify) → generate → cited
  answer. Done per request.

The project has **two query pipelines**:

- **Naive RAG** (`src/rag/naive_rag.py`) — the textbook baseline: retrieve →
  stuff into prompt → answer.
- **Agentic RAG** (`src/graph/`) — a stateful graph that grades, self-corrects,
  decomposes, and verifies. This is the "production" pipeline.

**Learning strategy:** understand naive RAG completely first, then treat every
graph node as "one specific improvement over naive."

---

## Part 1 — RAG fundamentals (concepts before code)

| Concept | What it means | Why it matters |
|---|---|---|
| **Embedding** | A model turns text into a vector (~1536 floats) capturing *meaning*. Similar meaning → nearby vectors. | Makes "semantic search" possible. Project uses OpenAI `text-embedding-3-small`. |
| **Vector store / index** | A DB optimized to find nearest vectors fast (ANN — Approximate Nearest Neighbor). | Project uses **ChromaDB**. Alternatives: FAISS, Pinecone, Weaviate, pgvector, Qdrant, Milvus. |
| **Cosine similarity** | Distance metric — the angle between two vectors. | How "relevance" is scored in dense retrieval. |
| **Chunking** | Splitting big docs into passages (~1000 chars). | LLMs have limited context; you retrieve *passages*, not whole docs. |
| **Top-k retrieval** | Fetch the k most similar chunks. | Too few → miss info; too many → noise + cost. Project default k=4. |
| **Context window** | Max tokens an LLM can read at once. | Governs how many chunks you can stuff in. |
| **Grounding / citations** | Forcing the answer to come *only* from retrieved sources, and naming them. | The core value prop: trustworthy, auditable answers. |
| **Hallucination** | LLM inventing plausible-but-false facts. | The enemy. Most "advanced" machinery exists to reduce this. |

**Keywords to own:** embedding, vector/dense retrieval, sparse retrieval, ANN,
cosine similarity, chunk/chunking, top-k, context window, prompt, grounding,
hallucination, citation, corpus.

---

## Part 2 — The ingestion pipeline (offline)

**Files:** `src/ingestion/loader.py`, `src/ingestion/chunker.py`,
`src/vectorstore/chroma_store.py`, driven by `scripts/ingest.py`.

Flow: `Loader → Chunker → Embeddings → ChromaDB`.

### 2.1 Loading (`loader.py`)
- Reads PDF/TXT/MD and attaches **metadata**: `source`, `filename`, `doc_type`,
  `department`, `access_level`.
- **Design decision:** `department` and `access_level` are inferred from the
  **folder structure** (`data/sample_docs/hr/handbook.md` → dept=hr). The
  filesystem *is* the taxonomy.
- **Concept:** metadata is as important as content — it powers filtered
  retrieval and (should power) access control.

### 2.2 Chunking (`chunker.py`)
- Uses **`RecursiveCharacterTextSplitter`** (size=1000, overlap=200),
  markdown-aware.
- **Why overlap?** So a sentence split across a boundary still appears whole in
  one chunk. Prevents "lost at the seam" retrieval misses.
- **Why recursive?** It splits on natural boundaries (paragraphs → sentences →
  words) before hard-cutting.
- **Topics:** fixed vs. recursive vs. semantic vs. document-structure-aware
  chunking; chunk-size/overlap tuning; "small chunks for precision, big for
  context."

### 2.3 Embedding + storing (`chroma_store.py`)
- Embeds each chunk and persists to ChromaDB.
- **Content-hash deduplication:** chunk ID = SHA-256 of
  (source + start_index + content). Re-ingesting the same file adds 0
  duplicates — **idempotent ingestion**.
- **Singleton pattern:** the embeddings client and vector store are created
  once, not per call — avoids expensive re-initialization.

**Keywords:** document loader, metadata, text splitter, chunk overlap,
idempotency, content hashing, singleton, persistence.

---

## Part 3 — Retrieval (the heart of RAG)

**Files:** `src/retrieval/` — `factory.py`, `hybrid.py`, `multi_query.py`,
`rerank.py`, `cross_encoder_rerank.py`, `composed.py`.

This is where accuracy is won or lost. The project uses a **strategy pattern**:
`get_retriever(strategy, k, filter)` returns a retriever; the caller doesn't
care which. New strategies plug in without touching callers.

### The strategies, simple to advanced

| Strategy | How it works | When to use |
|---|---|---|
| **dense** | Pure vector similarity (ChromaDB). | Fast, semantic, the baseline. |
| **sparse (BM25)** | Keyword frequency scoring (`rank_bm25`). Great for exact terms, IDs, acronyms. | Complements dense. |
| **hybrid** | Run dense **and** BM25, fuse with **RRF**. | **The workhorse.** Semantic + keyword recall. |
| **multi_query** | LLM rewrites the question into 3 variants, retrieve all, dedup. | Ambiguous/underspecified questions. |
| **rerank** | Over-fetch (12), then an LLM scores each 0–10, keep top-k. | High precision. |
| **cross_rerank** | Same but a dedicated **cross-encoder** model scores relevance. | Faster + higher quality than LLM scoring. |
| **hybrid_cross_rerank** | Hybrid recall → cross-encoder precision. | Maximum quality. |
| **auto** | Picks a strategy from the query's **intent** (`resolve_strategy`). | Hands-off. |

### Core retrieval concepts to master
- **Dense vs. sparse vs. hybrid.** Dense = meaning; sparse = exact words;
  hybrid = both. The single most important retrieval concept in industry today.
- **RRF (Reciprocal Rank Fusion):** a parameter-free way to merge two ranked
  lists — score = Σ 1/(k + rank). Used in `hybrid.py`. Standard, no tuning.
- **Reranking / two-stage retrieval:** *recall then precision.* Cheaply fetch
  many candidates, then expensively re-score a few. **Bi-encoder** (embeddings,
  compares vectors) vs. **cross-encoder** (reads query+doc *together*, much more
  accurate, slower). "Retrieve-then-rerank" is a production standard.
- **Query transformation:** rewriting the raw question for better retrieval
  (`query_transformer.py`, `normalizer.py`, `entity_extractor.py`). Related:
  **HyDE** (Hypothetical Document Embeddings).
- **Metadata filtering:** restrict retrieval by `department`/`access_level`
  (`dept_detector.py` auto-detects department).

**Keywords:** BM25, TF-IDF, RRF, bi-encoder, cross-encoder, reranking, two-stage
retrieval, recall vs. precision, query expansion, HyDE, MMR (maximal marginal
relevance), metadata filtering, strategy/factory pattern.

---

## Part 4 — Generation & grounding

**Files:** `src/rag/naive_rag.py`, `src/context/context_builder.py`, prompts in
`src/graph/nodes.py`.

- **LCEL (LangChain Expression Language):** the `prompt | llm | parser` pipe
  syntax — a declarative way to compose a chain. `naive_rag.py` is pure LCEL.
- **Context building (`context_builder.py`):** dedup retrieved chunks, group by
  source, and **budget tokens** so you don't overflow the context window. Format
  with filenames so the model can cite.
- **Grounding prompt:** "answer ONLY from context; if not present, say 'I don't
  have enough information.'"
- **Structured output:** the grader/critic use
  `with_structured_output(PydanticModel)` — the LLM returns a **typed object**
  (e.g. `GradeResult(relevant=True)`), not text you parse. A big reliability win
  and industry best practice.

**Keywords:** LCEL, prompt template, system/human message, few-shot examples,
structured output / function calling, token budgeting, "I don't know"
enforcement, temperature.

---

## Part 5 — From naive RAG to Agentic RAG (LangGraph + CRAG)

**Files:** `src/graph/state.py`, `build_graph.py`, `nodes.py`, `planner.py`.

Naive RAG has a flaw: if retrieval returns junk, the LLM answers from junk.
**Agentic RAG** adds a *control loop* that can inspect, correct, and verify.

### Key framework concepts
- **LangGraph `StateGraph`:** a graph where **nodes** are functions that
  read/update a shared **state** (`RAGState`, a `TypedDict`), and **edges**
  decide what runs next. Supports **cycles** (loops), which plain LCEL can't.
- **Conditional edges / routers:** functions like `decide_after_grade` return
  the *name* of the next node — how the graph branches and loops.
- **Checkpointer:** persists graph state between steps (in-memory by default
  here; SQLite optional). Enables resumable/multi-turn flows.

### CRAG (Corrective RAG) — the central pattern
The simple-question path:
```
retrieve → grade_documents → decide:
   relevant   → generate
   not enough → transform_query → retrieve   (corrective retry loop)
   exhausted  → web_search → generate         (fallback to Tavily)
generate → critic → END
```
- **Grade:** an LLM judges "are these docs actually relevant?" If no, don't
  answer from them.
- **Corrective loop:** rewrite the query and try again (bounded by a retry
  counter — the "loop guard").
- **Web fallback:** if the corpus can't answer after retries, search the web
  (Tavily).
- **Critic (self-verification):** after generating, extract each claim and
  verify it against the sources; **strip unsupported claims.** This is what
  pushes faithfulness up.

### Multi-agent decomposition (`planner.py`)
- **Planner:** classifies simple vs. multi-part; if multi-part, **decomposes**
  into sub-questions ("Compare X and Y" → answer X, answer Y).
- **Sub-query processing** (sequential or parallel) → **synthesize** into one
  cited answer.
- This is the **agentic** part: the system *plans* how to answer.

**Keywords:** agentic RAG, state machine, node/edge/router, conditional edge,
cycle, checkpointer, CRAG, self-correction/self-reflection, query rewriting,
web-search fallback, critic/verifier, self-consistency, query decomposition,
planner/synthesizer, multi-hop reasoning.

---

## Part 6 — The full graph, node by node (the actual runtime)

**File:** `src/graph/build_graph.py` — the map. Every node is feature-flagged
(compiled in only if enabled). The real order:

```
START
 → [guardrail_check]   # block prompt-injection / unsafe input
 → [load_memory]       # pull prior conversation turns
 → [cache_lookup]      # semantic cache — return instantly on a hit
 → scope_check         # in-domain? out-of-scope → polite refusal
 → [intent_detect] → [query_transform] → [tool_router] → planner
       (or, if UNIFIED_ANALYSIS=on: one analyze_query node replaces those three)
 → route_after_plan:
      simple    → retrieve → grade → (generate | transform→retry | web→generate)
      multi-part→ process_sub_query (loop) → synthesize
 → critic              # verify claims (always / adaptive / off)
 → [cache_store] → [save_memory] → END
```

Study each node as "one job":

| Node | File | Job |
|---|---|---|
| `guardrail_check` | `security/guardrails.py` | Input safety (injection, PII, length). |
| `load_memory`/`save_memory` | `graph/memory_nodes.py` | Multi-turn conversation context. |
| `cache_lookup`/`cache_store` | `graph/cache_nodes.py` | Semantic caching. |
| `scope_check` | `graph/scope_detector.py` | Domain gate — refuse off-topic. |
| `intent_detect` | `graph/intent_detector.py` | Classify into 6 intents (routes retrieval). |
| `query_transform` | `retrieval/query_transformer.py` | Normalize + rewrite + extract entities. |
| `analyze_query` | `graph/analyzer.py` | **Optimization:** all three above in one LLM call. |
| `planner` | `graph/planner.py` | Decompose multi-part questions. |
| `retrieve`/`grade_documents` | `graph/nodes.py` | Fetch + judge relevance. |
| `transform_query`/`web_search` | `graph/nodes.py` | Corrective retry + web fallback. |
| `generate`/`critic` | `graph/nodes.py` | Answer + verify. |

**Design lesson:** a chain of small, single-responsibility, independently
testable nodes — one file, one job, one test file.

---

## Part 7 — Advanced intelligence layers

| Layer | Files | Concept to learn |
|---|---|---|
| **Knowledge Graph RAG** | `src/knowledge_graph/` (NetworkX) | Extract entities+relationships → graph → multi-hop traversal. **GraphRAG.** |
| **Semantic cache** | `src/cache/semantic_cache.py` | Cache by *embedding similarity*, not exact string. |
| **Conversation memory** | `src/memory/` | Token-budgeted history so follow-ups work. |
| **Tools / MCP** | `src/tools/`, `src/mcp/` | Calculator, data-lookup; **MCP (Model Context Protocol)** = emerging standard for exposing tools to LLMs. |
| **Intent-routed retrieval** | `retrieval/factory.py` | Route strategy by query type. |

**Keywords:** GraphRAG, entity/relationship extraction, semantic caching,
conversational memory, tool use / function calling, MCP, agents, ReAct pattern.

---

## Part 8 — Production-grade concerns

### Security (`src/security/`)
- **Authentication** (`auth.py`): Bearer API keys, **constant-time comparison**
  (`secrets.compare_digest`) to prevent timing attacks; key → stable identity
  hash for session scoping.
- **Guardrails** (`guardrails.py`): input validation — prompt-injection
  patterns, length limits, PII.
- **Output filtering** (`output_filter.py`): redact PII from responses — applied
  on **every streamed token**, not just the final answer.
- **Path/upload safety** (`api/app.py`): ingest confined to a root; upload
  filename/extension/MIME/size checks (prevents path traversal &
  arbitrary-file-read).
- **Known gap to study:** the **RBAC gap** — `access_level` metadata exists but
  isn't enforced per-user. Real enterprise systems must scope retrieval to the
  caller's permissions.

### Resilience (`src/resilience/circuit_breaker.py`)
- **Circuit breaker:** after N failures, "open" the circuit and stop calling the
  failing service (fail fast), then "half-open" to test recovery.
- **Timeouts, retries, graceful degradation:** fall back to zero-LLM heuristics
  when the LLM is down.
- **Design lesson:** a grader *error* shouldn't be treated as "docs are bad" —
  proceed and let the critic catch problems.

### Observability (`src/observability/`)
- **Cost/token tracking** (`cost_callback.py`): a LangChain callback hooks
  `on_llm_end` to capture tokens and compute USD per query. **LLMOps
  essential.**
- **Metrics store** (`metrics_store.py`): SQLite persistence of per-query
  cost/latency.
- **Tracing** (`tracing.py`): per-node timing; plus optional **LangSmith** for
  full trace export.
- **Health checks** (`health_checker.py`): liveness + deep subsystem checks.

**Keywords:** authn/authz, RBAC, prompt injection, PII redaction, constant-time
comparison, path traversal, circuit breaker, timeout/retry/backoff, graceful
degradation, idempotency, observability, tracing, LLMOps, cost tracking,
structured logging, rate limiting.

---

## Part 9 — Serving: API, UI, Deployment

- **FastAPI** (`api/app.py`): async endpoints, dependency injection
  (`Depends(verify_api_key)`), Pydantic request/response models (`api/models.py`).
- **SSE streaming:** Server-Sent Events push tokens as they're generated. The
  `_iter_in_thread` bridge makes it *genuinely* incremental (sync generator →
  async queue, contextvars propagation).
- **Rate limiting** (`slowapi`): per-IP request caps; heavier limits on
  expensive endpoints.
- **Streamlit UI** (`ui/app.py`): thin client that only proxies to the API
  (least privilege — no direct key access).
- **Docker** (`Dockerfile.prod`, `docker-compose.prod.yml`): multi-stage build,
  **non-root user**, healthchecks, resource limits, single-worker rationale
  (process-local state).

**Keywords:** REST, async/await, dependency injection, Pydantic validation, SSE
vs. WebSockets, rate limiting, CORS, containerization, multi-stage build, least
privilege, horizontal scaling, stateless services.

---

## Part 10 — Evaluation (you can't improve what you can't measure)

**Files:** `src/eval/ragas_eval.py`, `eval_set.json`.

- **RAGAS** — the standard RAG eval framework. The **"RAG triad" / core
  metrics:**
  - **Faithfulness** — is the answer grounded in the retrieved context?
    (anti-hallucination)
  - **Answer Relevancy** — does it actually address the question?
  - **Context Precision** — are the retrieved chunks relevant? (retrieval
    quality)
  - **Context Recall** — did retrieval find *all* needed info?
- **Golden dataset** (`eval_set.json`): curated Q/A pairs with ground truth —
  including out-of-corpus questions to test honesty.
- **A/B evaluation:** the harness compares strategies (dense vs. hybrid vs.
  graph) with numbers — justifying design decisions instead of guessing.

**Keywords:** RAGAS, faithfulness, answer relevancy, context precision/recall,
golden/eval dataset, ground truth, LLM-as-judge, regression testing, A/B
testing, offline vs. online eval.

---

## Part 11 — The Master Study Checklist

Work top-to-bottom. ✅ = implemented in this repo (go read it).

**Fundamentals**
- [ ] Tokens, embeddings, vector spaces, cosine similarity
- [ ] LLM basics: context window, temperature, system/user prompts, function calling
- [ ] Prompt engineering: grounding, few-shot, "I don't know" enforcement ✅
- [ ] Chunking strategies (fixed/recursive/semantic) + overlap ✅
- [ ] Vector databases & ANN; ChromaDB vs. FAISS/pgvector/Pinecone/Qdrant ✅
- [ ] Naive RAG end-to-end ✅

**Core retrieval**
- [ ] Dense vs. sparse (BM25) vs. hybrid ✅
- [ ] RRF fusion ✅
- [ ] Reranking; bi-encoder vs. cross-encoder ✅
- [ ] Query transformation, multi-query, HyDE, MMR ✅ (first three)
- [ ] Metadata filtering & access scoping ✅

**Agentic / advanced RAG**
- [ ] LangChain (LCEL) vs. LangGraph (StateGraph) ✅
- [ ] Corrective RAG (grade → correct → verify) ✅
- [ ] Self-reflection / critic / self-consistency ✅
- [ ] Query decomposition + synthesis (multi-hop) ✅
- [ ] Agents, ReAct, tool use, MCP ✅
- [ ] GraphRAG (entities + relationships) ✅
- [ ] Semantic caching ✅
- [ ] Conversational memory ✅

**Production / LLMOps**
- [ ] Structured output (Pydantic) ✅
- [ ] Guardrails, prompt-injection defense, PII redaction ✅
- [ ] AuthN/AuthZ, RBAC (⚠️ partial — study the gap) ✅
- [ ] Circuit breakers, timeouts, retries, graceful degradation ✅
- [ ] Observability: tracing, cost/token tracking, metrics ✅
- [ ] Rate limiting, CORS, secrets management ✅
- [ ] Evaluation with RAGAS + golden datasets ✅
- [ ] Testing: unit/integration/e2e, mocking LLMs, test isolation ✅
- [ ] Streaming (SSE), async, concurrency ✅
- [ ] Docker, multi-stage builds, non-root, healthchecks ✅
- [ ] CI/CD, dependency pinning (⚠️ study what's missing)

**Frontier topics (explore beyond this repo)**
- [ ] Late-interaction retrieval (ColBERT), SPLADE (learned sparse)
- [ ] Contextual retrieval (chunk-context prepending)
- [ ] Long-context vs. RAG trade-offs; RAG + long context
- [ ] Fine-tuning vs. RAG; embedding fine-tuning
- [ ] Agent frameworks & multi-agent orchestration
- [ ] Vector index types (HNSW, IVF), quantization
- [ ] Guardrail frameworks (NeMo Guardrails, Llama Guard)
- [ ] Semantic chunking, proposition-based indexing, RAPTOR (hierarchical)

---

## Part 12 — How to build this from scratch (suggested path)

1. **Week 1 — Naive RAG:** load → chunk → embed → Chroma → LCEL chain with a
   grounding prompt. Ask questions. *You now have working RAG.*
2. **Week 2 — Better retrieval:** add BM25, hybrid + RRF, then a cross-encoder
   reranker. Measure the lift with a tiny eval set.
3. **Week 3 — Agentic:** move to LangGraph; add grade → corrective-retry →
   generate. Add the critic. Add the planner for multi-part questions.
4. **Week 4 — Production:** FastAPI + streaming, auth, guardrails, rate limiting,
   circuit breaker, cost tracking, Docker.
5. **Ongoing — Evaluate:** RAGAS from day one; let numbers drive every choice.

**The meta-lesson:** start simple, make each improvement *one small,
independently-testable, feature-flagged component*, and **measure everything.**
That's how you go from a toy to a production RAG system.

---

## Where to read next (file tour order)

1. `config.py` — every knob in one place.
2. `src/ingestion/loader.py` → `chunker.py` → `src/vectorstore/chroma_store.py`.
3. `src/rag/naive_rag.py` — the whole idea in one file.
4. `src/retrieval/factory.py` → `hybrid.py` → `cross_encoder_rerank.py`.
5. `src/graph/state.py` → `build_graph.py` → `nodes.py` → `planner.py`.
6. `src/security/`, `src/resilience/`, `src/observability/`.
7. `api/app.py` → `ui/app.py`.
8. `src/eval/ragas_eval.py` — close the loop with measurement.

Read the matching `tests/test_*.py` for each — the tests show exactly how each
component is meant to be used.
