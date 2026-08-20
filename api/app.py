"""FastAPI application for the Enterprise RAG Assistant.

Endpoints:
    GET  /health              — Liveness check with collection stats
    POST /ask                 — Query the RAG pipeline (supports streaming via SSE)
    POST /ingest              — Trigger server-side document ingestion
    POST /upload              — Upload a document (202 + async indexing when enabled)
    GET  /documents           — List documents and their ingestion status
    GET  /documents/{id}      — Status of one uploaded document
    POST /eval                — Run RAGAS evaluation suite
    GET  /tools               — List MCP tool registry
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import (
    Depends,
    FastAPI,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from slowapi.errors import RateLimitExceeded
from starlette.responses import JSONResponse

from api.models import (
    AskRequest,
    AskResponse,
    DocumentListResponse,
    DocumentRetryResponse,
    DocumentStatusResponse,
    ErrorResponse,
    EvalRequest,
    EvalResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    UploadAcceptedResponse,
    UploadResponse,
)
from api.rate_limit import build_limiter, verify_storage
from config import settings, setup_logging
from src.ingestion.backpressure import (
    QueueSaturated,
    check_capacity,
    retry_after_seconds,
)
from src.observability import tracing_otel as otel
from src.observability.cost_guard import daily_cap_exceeded
from src.observability.request_context import (
    get_request_id,
    new_request_id,
    reset_request_id,
    set_request_id,
)
from src.security.access_control import (
    CONFIDENTIAL_DEPARTMENTS,
    DepartmentForbidden,
    enforce_scope,
)
from src.security.auth import permitted_departments, verify_api_key
from src.security.guardrails import check_guardrails
from src.security.output_filter import filter_output
from src.storage.sql import warn_if_single_node

logger = logging.getLogger(__name__)

VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_error_detail(e: Exception) -> str:
    """Return error detail safe for client consumption.

    In debug mode, returns the full exception message.
    In production, returns a generic message to avoid leaking internals.
    """
    if settings.debug_mode:
        return str(e)
    return "An internal error occurred. Please try again or contact support."


def _seconds_until_utc_midnight() -> int:
    """Retry-After for the daily cost cap, which resets at UTC midnight."""
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC)
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1, int((tomorrow - now).total_seconds()))


def _scoped_session_id(request: Request, session_id: str | None) -> str | None:
    """Bind a client-supplied session ID to the caller's API key identity.

    Without this, any caller who guesses another user's session ID gets
    their conversation history injected into the prompt. When auth is
    disabled (single-tenant dev mode) the session ID is used as-is.
    """
    if not session_id:
        return None
    owner = getattr(request.state, "api_key_id", "")
    if owner:
        return f"{owner}:{session_id}"
    return session_id


async def _iter_in_thread(sync_iter_factory, request: Request):
    """Bridge a synchronous generator into an async one, item by item.

    Runs the sync iteration in a worker thread (asyncio.to_thread copies
    contextvars, so LangChain callbacks and tracing captures propagate)
    and pushes items onto a bounded queue as they are produced — this is
    what makes SSE genuinely incremental instead of a burst replay at the
    end. Stops the producer between items when the client disconnects.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=32)
    stop = threading.Event()

    def _produce():
        try:
            for item in sync_iter_factory():
                if stop.is_set():
                    break
                asyncio.run_coroutine_threadsafe(queue.put(("item", item)), loop).result()
            asyncio.run_coroutine_threadsafe(queue.put(("end", None)), loop).result()
        except Exception as exc:  # surfaced to the consumer below
            with contextlib.suppress(Exception):
                asyncio.run_coroutine_threadsafe(queue.put(("error", exc)), loop).result()

    producer = asyncio.create_task(asyncio.to_thread(_produce))
    try:
        while True:
            try:
                kind, item = await asyncio.wait_for(queue.get(), timeout=1.0)
            except TimeoutError:
                if await request.is_disconnected():
                    return
                continue
            if kind == "end":
                return
            if kind == "error":
                raise item
            yield item
            if await request.is_disconnected():
                return
    finally:
        stop.set()
        # Drain so a producer blocked on queue.put can observe the stop flag.
        while not queue.empty():
            queue.get_nowait()
        with contextlib.suppress(Exception):
            await producer

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

limiter = build_limiter()


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings.validate()
    logger.info("Enterprise RAG Assistant API starting (v%s)", VERSION)

    # Surface degraded file-type support at boot, not on the first upload.
    from src.ingestion.loader import SUPPORTED_SUFFIXES, available_suffixes, missing_dependency

    unavailable = sorted(set(SUPPORTED_SUFFIXES) - available_suffixes())
    if unavailable:
        logger.warning(
            "File types unavailable on this deployment (uploads will be rejected): %s",
            ", ".join(f"{s} (needs {missing_dependency(s)})" for s in unavailable),
        )
    logger.info("Accepting uploads for: %s", ", ".join(sorted(available_suffixes())))

    # Report retrieval scoping. An unscoped key can read every
    # department including legal and security, so a deployment that
    # believes it is segmented should be able to see that it is not.
    if settings.auth_enabled:
        from src.security.auth import _get_key_scopes

        scopes = _get_key_scopes()
        unscoped = sum(1 for v in scopes.scopes.values() if v is None)
        if unscoped:
            logger.warning(
                "%d of %d API key(s) are unscoped and can retrieve from every "
                "department, including %s. Scope them with "
                "API_KEYS=<key>:hr|general.",
                unscoped,
                len(scopes.scopes),
                ", ".join(sorted(CONFIDENTIAL_DEPARTMENTS)),
            )
        if scopes.any_scoped:
            logger.info("Department scoping active on %d API key(s)", len(scopes.scopes) - unscoped)

    # Traces are optional; this is a no-op unless OTEL_ENABLED=true.
    otel.setup_tracing("rag-api")

    # Local SQLite files look identical to a working deployment right up
    # until a second replica exists.
    warn_if_single_node()

    # Probe the rate limiter's backend now. A shared store that is
    # unreachable degrades to per-process counters, which looks healthy
    # but silently multiplies the effective limit by the replica count.
    verify_storage(limiter)
    yield
    logger.info("API shutting down")


app = FastAPI(
    title="Enterprise RAG Assistant",
    description="AI-powered question answering over enterprise documents with source citations.",
    version=VERSION,
    lifespan=lifespan,
)

app.state.limiter = limiter


# Rate limit error handler
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"error": "Rate limit exceeded", "detail": str(exc.detail)},
    )


# CORS — configurable via env vars
_cors_origins = [o.strip() for o in settings.cors_origins.split(",")]
_cors_methods = [m.strip() for m in settings.cors_allow_methods.split(",")]
_cors_headers = [h.strip() for h in settings.cors_allow_headers.split(",")]
if "*" in _cors_origins:
    logger.warning("CORS_ORIGINS is set to '*' — all origins allowed. Restrict for production.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials="*" not in _cors_origins,
    allow_methods=_cors_methods,
    allow_headers=_cors_headers,
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Bind a request id for the lifetime of the request and echo it back.

    Honours a caller-supplied X-Request-ID so a trace started upstream (a
    gateway, the UI) stays intact. The id is stamped onto every log line
    and onto published ingestion events, which is what lets a single
    upload be followed across the API, the queue and the worker.
    """
    incoming = request.headers.get("X-Request-ID", "").strip()
    request_id = incoming[:64] if incoming else new_request_id()
    token = set_request_id(request_id)
    request.state.request_id = request_id
    try:
        response = await call_next(request)
    finally:
        reset_request_id(token)
    response.headers["X-Request-ID"] = request_id
    return response


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------

@app.get("/metrics", response_model=None, responses={401: {"model": ErrorResponse}})
async def metrics_endpoint(request: Request):
    """Prometheus exposition of cost, latency, quality and queue metrics.

    Authenticated for the same reason ``/health?deep=true`` is: the body
    carries queue depths, document counts and per-department totals.
    Prometheus scrapes it with a bearer token from its scrape config.

    Served as text/plain — that is the exposition format, and returning
    JSON here would simply not be scrapeable.
    """
    await verify_api_key(request)

    from src.observability.prometheus import render

    body = await asyncio.to_thread(render)
    return Response(content=body, media_type="text/plain; version=0.0.4; charset=utf-8")


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health", response_model=None, responses={401: {"model": ErrorResponse}})
async def health(request: Request, deep: bool = False):
    """Liveness check with collection stats. Pass ?deep=true for subsystem checks.

    The shallow response stays unauthenticated so load balancers and
    container orchestrators can probe it. ``?deep=true`` requires a valid
    API key: it reports the vector store endpoint, queue depths, registry
    counts and process memory — internal topology that should not be
    readable by anything that can merely reach the port.
    """
    if deep:
        await verify_api_key(request)

        from src.observability.health_checker import deep_health_check
        result = deep_health_check()
        return JSONResponse(content={
            "status": result.status,
            "version": VERSION,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status,
                    "latency_ms": round(c.latency_ms, 1),
                    "detail": c.detail,
                }
                for c in result.checks
            ],
        })
    try:
        from src.vectorstore.chroma_store import collection_stats
        stats = collection_stats()
        return HealthResponse(
            status="ok",
            collection=stats["collection"],
            document_count=stats["document_count"],
            version=VERSION,
        )
    except Exception:
        return HealthResponse(
            status="degraded",
            collection=settings.chroma_collection,
            document_count=-1,
            version=VERSION,
        )


# ---------------------------------------------------------------------------
# POST /ask
# ---------------------------------------------------------------------------

def _resolve_mode(body: AskRequest) -> str:
    """Resolve 'auto' mode to either 'naive' or 'graph'."""
    if body.mode != "auto":
        return body.mode
    q = body.question.strip().lower()
    # Heuristic: use graph for complex queries
    word_count = len(q.split())
    has_comparison = any(w in q for w in ["compare", "versus", "vs", "difference between", "contrast"])
    has_multi = any(w in q for w in [" and ", "both", "also", "as well as", "additionally"])
    if word_count > 15 or has_comparison or has_multi:
        return "graph"
    return "naive"


def _ask_sync(body: AskRequest, session_id: str | None = None) -> AskResponse:
    """Run the query synchronously and return a full AskResponse.

    *session_id* is the already-scoped session identifier (bound to the
    caller's API key by the endpoint).
    """
    from src.observability.cost_callback import CostCallbackHandler, is_idk_response
    from src.observability.cost_guard import start_query_budget

    resolved_mode = _resolve_mode(body)
    # Ceiling for this query. Nodes consult it before optional, LLM-heavy
    # work so a pathological question degrades instead of fanning out.
    start_query_budget()
    handler = CostCallbackHandler()
    start = time.perf_counter()
    node_latencies = None

    if resolved_mode == "graph":
        from src.graph.build_graph import get_graph
        from src.graph.tracing import start_run_capture

        graph = get_graph()
        import uuid
        tid = uuid.uuid4().hex[:12]
        sid = session_id or tid
        config = {"configurable": {"thread_id": tid}, "callbacks": [handler]}
        invoke_state = {
            "question": body.question,
            "retries": 0,
            "retriever_strategy": body.retriever_strategy,
            "session_id": sid,
        }
        if body.filter:
            invoke_state["filter"] = body.filter
        if body.top_k:
            invoke_state["top_k"] = body.top_k
        start_run_capture()  # scope node latencies to this request
        result = graph.invoke(invoke_state, config)
        answer = filter_output(result.get("generation", "No answer was generated."))

        # Capture per-node latencies from tracing
        try:
            from src.graph.tracing import get_last_run_latencies
            node_latencies = get_last_run_latencies()
        except Exception:
            pass
    else:
        from src.rag.naive_rag import build_naive_rag_chain

        chain = build_naive_rag_chain(
            k=body.top_k,
            filter=body.filter,
            retriever_strategy=body.retriever_strategy,
        )
        answer = filter_output(chain.invoke(body.question, config={"callbacks": [handler]}))

    latency_ms = (time.perf_counter() - start) * 1000
    is_idk = is_idk_response(answer)
    metrics = handler.flush(
        thread_id="api",
        question=body.question,
        latency_ms=latency_ms,
        retriever_strategy=body.retriever_strategy,
        mode=resolved_mode,
        is_idk=is_idk,
        node_latencies=node_latencies,
    )

    # Record to metrics store (best-effort)
    try:
        from src.observability.metrics_store import get_store
        get_store().record(metrics)
    except Exception:
        logger.debug("Failed to record API query metrics", exc_info=True)

    return AskResponse(
        answer=answer,
        question=body.question,
        mode=resolved_mode,
        retriever_strategy=body.retriever_strategy,
        cost_usd=metrics.estimated_cost_usd,
        latency_ms=latency_ms,
        tokens_used=metrics.total_tokens,
        node_latencies=node_latencies,
        is_idk=is_idk,
        # Echo the client's own session id, not the internally scoped one.
        session_id=body.session_id,
    )


async def _stream_graph(body: AskRequest, request: Request, session_id: str | None = None):
    """Async generator yielding SSE events for graph mode streaming.

    Steps are forwarded to the client as the pipeline produces them (via
    _iter_in_thread), and the producer is stopped between steps if the
    client disconnects. All streamed content passes through the PII
    output filter before leaving the server.
    """
    try:
        from src.graph.tracing import start_run_capture
        from src.observability.cost_callback import CostCallbackHandler
        from src.observability.cost_guard import start_query_budget

        start_query_budget()
        handler = CostCallbackHandler()
        start = time.perf_counter()

        import uuid

        from src.graph.build_graph import get_graph

        graph = get_graph()
        tid = uuid.uuid4().hex[:12]
        sid = session_id or tid
        config = {"configurable": {"thread_id": tid}, "callbacks": [handler]}

        answer = ""
        stream_state = {
            "question": body.question,
            "retries": 0,
            "retriever_strategy": body.retriever_strategy,
            "session_id": sid,
        }
        if body.filter:
            stream_state["filter"] = body.filter
        if body.top_k:
            stream_state["top_k"] = body.top_k

        start_run_capture()  # scope node latencies to this request
        async for step in _iter_in_thread(
            lambda: graph.stream(stream_state, config), request
        ):
            for node_name, state_update in step.items():
                event = {"type": "status", "node": node_name}
                if state_update and "generation" in state_update:
                    # Redact PII before it leaves the server — filtering
                    # only the final answer would leak via token events.
                    answer = filter_output(state_update["generation"])
                    event["type"] = "token"
                    event["content"] = answer
                yield f"data: {json.dumps(event)}\n\n"

        latency_ms = (time.perf_counter() - start) * 1000
        disconnected = await request.is_disconnected()
        if disconnected:
            logger.info("Client disconnected during streaming (thread=%s)", tid)

        # Capture per-node latencies and IDK status
        from src.observability.cost_callback import is_idk_response
        node_latencies = None
        try:
            from src.graph.tracing import get_last_run_latencies
            node_latencies = get_last_run_latencies()
        except Exception:
            pass

        is_idk = is_idk_response(answer)
        metrics = handler.flush(
            thread_id=tid, question=body.question,
            latency_ms=latency_ms, retriever_strategy=body.retriever_strategy,
            mode="graph",
            is_idk=is_idk,
            node_latencies=node_latencies,
        )

        try:
            from src.observability.metrics_store import get_store
            get_store().record(metrics)
        except Exception:
            pass

        if disconnected:
            return  # cost is recorded above; no one is listening for events

        done_event = {
            "type": "done",
            "answer": answer,
            "cost_usd": metrics.estimated_cost_usd,
            "latency_ms": latency_ms,
            "tokens_used": metrics.total_tokens,
            "node_latencies": node_latencies,
            "is_idk": is_idk,
        }
        yield f"data: {json.dumps(done_event)}\n\n"
    except Exception as e:
        logger.exception("Streaming graph error")
        err = {"type": "error", "message": _safe_error_detail(e)}
        yield f"data: {json.dumps(err)}\n\n"


async def _stream_naive(body: AskRequest, request: Request):
    """Async generator yielding SSE events for naive mode streaming.

    Chunks are forwarded as the LLM produces them (via _iter_in_thread).
    Each chunk passes through the PII output filter; patterns spanning a
    chunk boundary are additionally caught by the filtered final answer
    in the 'done' event.
    """
    from src.observability.cost_callback import CostCallbackHandler
    from src.observability.cost_guard import start_query_budget
    from src.rag.naive_rag import build_naive_rag_chain

    start_query_budget()
    handler = CostCallbackHandler()
    start = time.perf_counter()

    chain = build_naive_rag_chain(
        k=body.top_k,
        filter=body.filter,
        retriever_strategy=body.retriever_strategy,
    )

    parts: list[str] = []
    async for chunk in _iter_in_thread(
        lambda: chain.stream(body.question, config={"callbacks": [handler]}), request
    ):
        parts.append(chunk)
        event = {"type": "token", "content": filter_output(chunk)}
        yield f"data: {json.dumps(event)}\n\n"

    if await request.is_disconnected():
        logger.info("Client disconnected during naive streaming")
        return

    full_answer = filter_output("".join(parts))
    latency_ms = (time.perf_counter() - start) * 1000
    from src.observability.cost_callback import is_idk_response
    is_idk = is_idk_response(full_answer)
    metrics = handler.flush(
        thread_id="api", question=body.question,
        latency_ms=latency_ms, retriever_strategy=body.retriever_strategy,
        mode="naive",
        is_idk=is_idk,
    )

    try:
        from src.observability.metrics_store import get_store
        get_store().record(metrics)
    except Exception:
        pass

    done_event = {
        "type": "done",
        "answer": full_answer,
        "cost_usd": metrics.estimated_cost_usd,
        "latency_ms": latency_ms,
        "tokens_used": metrics.total_tokens,
        "is_idk": is_idk,
    }
    yield f"data: {json.dumps(done_event)}\n\n"


@app.post("/ask", response_model=AskResponse, responses={400: {"model": ErrorResponse}},
           dependencies=[Depends(verify_api_key)])
@limiter.limit(settings.rate_limit_per_minute)
async def ask_endpoint(request: Request, response: Response, body: AskRequest):
    """Query the RAG pipeline. Set stream=true for Server-Sent Events."""
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # API-layer guardrail check (covers both graph and naive modes)
    guardrail = check_guardrails(body.question)
    if not guardrail.safe:
        raise HTTPException(status_code=400, detail=guardrail.reason)

    # Daily spend ceiling. Unlike the per-query budget — which degrades
    # the answer — this one denies: its entire purpose is to stop runaway
    # spend, so past the cap an operator has to intervene.
    if daily_cap_exceeded():
        raise HTTPException(
            status_code=503,
            detail="Daily cost cap reached. Queries resume at the next UTC day.",
            headers={"Retry-After": str(_seconds_until_utc_midnight())},
        )

    # Bind the session to the caller's API key so one client cannot read
    # another client's conversation history by guessing session IDs.
    scoped_session = _scoped_session_id(request, body.session_id)

    # Constrain retrieval to the departments this key may read. Applied to
    # the request body so every downstream path — sync, streaming graph
    # and streaming naive — receives the same enforced filter, rather than
    # each having to remember to check.
    try:
        body.filter = enforce_scope(body.filter, permitted_departments(request))
    except DepartmentForbidden as e:
        logger.warning(
            "Blocked cross-department retrieval: key=%s requested=%s allowed=%s",
            getattr(request.state, "api_key_id", "?"),
            sorted(e.requested),
            sorted(e.allowed),
        )
        raise HTTPException(status_code=403, detail=str(e)) from e

    try:
        if body.stream:
            resolved = _resolve_mode(body)
            gen = (
                _stream_graph(body, request, session_id=scoped_session)
                if resolved == "graph"
                else _stream_naive(body, request)
            )
            return StreamingResponse(gen, media_type="text/event-stream")

        # Run sync pipeline in a thread to avoid blocking the event loop
        return await asyncio.to_thread(_ask_sync, body, scoped_session)
    except Exception as e:
        logger.exception("Ask endpoint failed")
        raise HTTPException(status_code=500, detail=_safe_error_detail(e)) from e


# ---------------------------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------------------------

def _ingest_sync(body: IngestRequest) -> IngestResponse:
    """Run ingestion synchronously (called via asyncio.to_thread)."""
    from src.ingestion.chunker import chunk_documents
    from src.ingestion.loader import load_path
    from src.vectorstore.chroma_store import add_chunks, collection_stats

    docs = load_path(body.path)
    chunks = chunk_documents(
        docs,
        chunk_size=body.chunk_size,
        chunk_overlap=body.chunk_overlap,
    )
    added = add_chunks(chunks)
    stats = collection_stats()

    return IngestResponse(
        documents_loaded=len(docs),
        chunks_created=len(chunks),
        chunks_added=added,
        collection_total=stats["document_count"],
    )


def _validate_ingest_path(raw_path: str) -> None:
    """Reject paths outside the configured ingest root.

    Without this check, /ingest is an arbitrary-file-read primitive: any
    .pdf/.txt/.md on the server could be ingested and then queried.
    """
    root = Path(settings.ingest_root).resolve()
    target = Path(raw_path).resolve()
    if target != root and root not in target.parents:
        raise HTTPException(
            status_code=400,
            detail=f"Path must be inside the ingest root ({settings.ingest_root}).",
        )


@app.post("/ingest", response_model=IngestResponse, responses={400: {"model": ErrorResponse}},
           dependencies=[Depends(verify_api_key)])
@limiter.limit(settings.heavy_rate_limit)
async def ingest_endpoint(request: Request, response: Response, body: IngestRequest):
    """Ingest documents from a file or directory path under the ingest root."""
    _validate_ingest_path(body.path)
    try:
        return await asyncio.to_thread(_ingest_sync, body)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Ingest endpoint failed")
        raise HTTPException(status_code=500, detail=_safe_error_detail(e)) from e


# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".csv", ".txt", ".md"}
ALLOWED_MIME_TYPES = {
    "application/pdf", "text/plain", "text/markdown", "text/x-markdown",
    "text/csv", "application/csv",
    "application/vnd.ms-excel",  # what several browsers send for .csv
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
    "application/octet-stream",  # browsers often send this for .md files
}


def _sanitize_filename(name: str) -> str:
    """Extract the base filename and reject unsafe characters."""
    # Strip directory components (prevents path traversal)
    base = Path(name).name
    if not base or ".." in base or "\x00" in base:
        raise ValueError(f"Invalid filename: {name!r}")
    return base


def _upload_sync(content: bytes, safe_name: str, department: str) -> UploadResponse:
    """Parse, chunk and index the upload inline (ASYNC_INGESTION=false).

    Runs on a worker thread: every call below is blocking I/O, and holding
    the event loop for the length of an embedding run would stall every
    concurrent request on the process.
    """
    from src.ingestion.chunker import chunk_documents
    from src.ingestion.loader import load_path
    from src.vectorstore.chroma_store import add_chunks, collection_stats

    # Save uploaded file to a temp directory so the loader can read it
    with tempfile.TemporaryDirectory() as tmpdir:
        dept_dir = Path(tmpdir) / department
        dept_dir.mkdir()
        dest = dept_dir / safe_name
        dest.write_bytes(content)
        logger.info(
            "Upload: saved %s (%d bytes) to temp dir, department=%s",
            safe_name, len(content), department,
        )

        docs = load_path(tmpdir)

        # Rewrite source metadata: replace temp path with a stable
        # identifier so citations are meaningful and content-hash
        # deduplication works across re-uploads of the same file.
        from src.ingestion.pipeline import stable_source

        source = stable_source(department, safe_name)
        for doc in docs:
            doc.metadata["source"] = source

        chunks = chunk_documents(docs)
        added = add_chunks(chunks)
        stats = collection_stats()

    logger.info(
        "Upload complete: %s -> %d docs, %d chunks, %d new (total %d)",
        safe_name, len(docs), len(chunks), added, stats["document_count"],
    )
    return UploadResponse(
        filename=safe_name,
        documents_loaded=len(docs),
        chunks_created=len(chunks),
        chunks_added=added,
        collection_total=stats["document_count"],
    )


def _upload_async(
    content: bytes,
    safe_name: str,
    department: str,
    content_type: str,
    uploaded_by: str,
    request_id: str,
) -> tuple[UploadAcceptedResponse, int]:
    """Store durably, queue for indexing, and return (response, status_code).

    This is the top lane of the ingestion architecture: idempotency check,
    generate document id if new, persist the bytes, publish the event,
    respond 202. No parsing or embedding happens on the request path.
    """
    from src.events.bus import DOCUMENT_UPLOADED, Event, get_event_bus
    from src.ingestion.registry import (
        STATUS_DEAD_LETTER,
        STATUS_FAILED,
        STATUS_PENDING,
        compute_checksum,
        get_registry,
    )
    from src.storage.object_store import build_storage_key, get_object_store

    checksum = compute_checksum(content)
    registry = get_registry()

    # --- Idempotency check / Generate Document ID (if new) ---
    record, created = registry.register_upload(
        checksum=checksum,
        filename=safe_name,
        department=department,
        content_type=content_type,
        size_bytes=len(content),
        uploaded_by=uploaded_by,
        request_id=request_id,
    )

    should_queue = created
    if not created:
        if record.status in (STATUS_FAILED, STATUS_DEAD_LETTER):
            # Re-uploading a document whose indexing gave up is an explicit
            # request to try again — reset the retry budget and requeue.
            should_queue = registry.reset_for_retry(record.document_id)
        elif record.status == STATUS_PENDING and not record.storage_key:
            # Registered but never stored: the previous attempt died between
            # the INSERT and the upload. Heal it instead of stranding it.
            logger.warning(
                "Document %s was registered without stored bytes — requeueing",
                record.document_id,
            )
            should_queue = True

    if not should_queue:
        # --- "Document Already Exists?" -> Yes -> Return Existing Document ID ---
        logger.info(
            "Duplicate upload of %s (%s) — returning existing document %s (status=%s)",
            safe_name, department, record.document_id, record.status,
        )
        return (
            UploadAcceptedResponse(
                document_id=record.document_id,
                filename=record.filename,
                department=record.department,
                status=record.status,
                duplicate=True,
                checksum=record.checksum,
                size_bytes=record.size_bytes,
                storage_uri=record.storage_uri,
                status_url=f"/documents/{record.document_id}",
                message="This document was already accepted; no new work was queued.",
            ),
            200,
        )

    document_id = record.document_id
    try:
        # --- Upload File to S3 ---
        store = get_object_store()
        storage_key = build_storage_key(department, document_id, safe_name)
        storage_uri = store.put(storage_key, content, content_type)
        registry.attach_storage(document_id, storage_key, storage_uri)

        # --- Publish Kafka Event ---
        # The trace context rides inside the payload. Indexing happens in
        # another process, possibly minutes later, so this is the only
        # carrier available — and without it the upload span and the
        # indexing span are two unrelated traces.
        with otel.span(
            "ingestion.publish",
            **{"document.id": document_id, "document.department": department},
        ):
            get_event_bus().publish(
                settings.kafka_topic_ingestion,
                Event(
                    event_type=DOCUMENT_UPLOADED,
                    document_id=document_id,
                    payload=otel.inject_context({
                        "storage_key": storage_key,
                        "filename": safe_name,
                        "department": department,
                        "checksum": checksum,
                    }),
                    request_id=request_id,
                ),
            )
    except Exception as e:
        # Leave a durable trace of the failure. The document is now FAILED,
        # so a re-upload takes the reset-and-requeue branch above rather
        # than being mistaken for an already-accepted duplicate.
        registry.mark_failed(document_id, f"{type(e).__name__}: {e}")
        logger.exception("Failed to enqueue document %s for ingestion", document_id)
        raise

    logger.info(
        "Accepted %s (%d bytes) as document %s — queued for indexing",
        safe_name, len(content), document_id,
    )
    return (
        UploadAcceptedResponse(
            document_id=document_id,
            filename=safe_name,
            department=department,
            status=STATUS_PENDING,
            duplicate=False,
            checksum=checksum,
            size_bytes=len(content),
            storage_uri=storage_uri,
            status_url=f"/documents/{document_id}",
            message="Accepted for indexing. Poll the status URL until status is PROCESSED.",
        ),
        202,
    )


@app.post(
    "/upload",
    response_model=None,
    responses={
        200: {"model": UploadResponse, "description": "Indexed inline (ASYNC_INGESTION=false) or duplicate"},
        202: {"model": UploadAcceptedResponse, "description": "Stored and queued for indexing"},
        400: {"model": ErrorResponse},
    },
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(settings.heavy_rate_limit)
async def upload_endpoint(
    request: Request,
    response: Response,
    file: UploadFile,
    department_form: str | None = Form(None, alias="department"),
    department_query: str | None = Query(None, alias="department"),
):
    """Upload a document (PDF, DOCX, CSV, TXT or MD) for indexing.

    With ASYNC_INGESTION=true the file is stored durably and queued, and
    this returns 202 immediately — poll GET /documents/{document_id} for
    progress. Otherwise the document is parsed and indexed inline and the
    chunk counts are returned.

    *department* is accepted as either a multipart form field or a query
    parameter. It must be declared both ways: a bare ``str`` default makes
    FastAPI read it from the query string only, so the form field browsers
    and the Streamlit UI actually send was being silently discarded and
    every upload filed itself under "general" — which also meant legal and
    security documents never received their "confidential" access level.
    """
    from api.models import VALID_DEPARTMENTS

    department = department_form or department_query or "general"

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    # --- Filename sanitization ---
    try:
        safe_name = _sanitize_filename(file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid filename.") from e

    # --- Extension check ---
    from src.ingestion.loader import available_suffixes, missing_dependency

    suffix = Path(safe_name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(available_suffixes()))}",
        )

    # A known type whose parser dependency is not installed. Reject it here
    # rather than accepting the upload and dead-lettering it moments later —
    # asynchronous failure hides the reason from the caller.
    package = missing_dependency(suffix)
    if package:
        logger.error(
            "Rejected %s upload: '%s' support requires the %s package", suffix, suffix, package
        )
        raise HTTPException(
            status_code=415,
            detail=(
                f"'{suffix}' files cannot be processed by this server: the {package} "
                f"package is not installed. Currently accepted: "
                f"{', '.join(sorted(available_suffixes()))}"
            ),
        )

    # --- MIME type check ---
    content_type = file.content_type or "application/octet-stream"
    if content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type '{content_type}'.",
        )

    # --- Department validation ---
    department = department.strip().lower()
    if department not in VALID_DEPARTMENTS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid department '{department}'. Allowed: {', '.join(sorted(VALID_DEPARTMENTS))}",
        )

    # A key scoped to a department must not be able to file documents into
    # another one. Without this, write is the way around read scoping:
    # upload into `legal`, then read it back through the legal filter.
    allowed_departments = permitted_departments(request)
    if allowed_departments is not None and department not in allowed_departments:
        logger.warning(
            "Blocked cross-department upload: key=%s department=%s allowed=%s",
            getattr(request.state, "api_key_id", "?"),
            department,
            sorted(allowed_departments),
        )
        raise HTTPException(
            status_code=403,
            detail=f"Not permitted for department(s): {department}",
        )

    # --- File size check ---
    content = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content)} bytes). Maximum: {settings.max_upload_size_mb} MB.",
        )

    # Refuse before storing the bytes, not after: accepting into a
    # saturated queue costs object storage and hands the caller a 202 for
    # work that will not happen for hours. Only meaningful on the async
    # path — the sync path does the work inline, so there is no backlog
    # to grow.
    if settings.async_ingestion:
        try:
            await asyncio.to_thread(check_capacity)
        except QueueSaturated as e:
            raise HTTPException(
                status_code=503,
                detail=str(e),
                headers={"Retry-After": str(retry_after_seconds(e.depth, e.limit))},
            ) from e

    try:
        if settings.async_ingestion:
            accepted, status_code = await asyncio.to_thread(
                _upload_async,
                content,
                safe_name,
                department,
                content_type,
                getattr(request.state, "api_key_id", ""),
                get_request_id(),
            )
            response.status_code = status_code
            return accepted

        return await asyncio.to_thread(_upload_sync, content, safe_name, department)
    except Exception as e:
        logger.exception("Upload endpoint failed for file: %s", safe_name)
        raise HTTPException(status_code=500, detail=_safe_error_detail(e)) from e


# ---------------------------------------------------------------------------
# GET /documents — ingestion status
# ---------------------------------------------------------------------------

def _to_status_response(record) -> DocumentStatusResponse:
    return DocumentStatusResponse(
        document_id=record.document_id,
        filename=record.filename,
        department=record.department,
        status=record.status,
        attempts=record.attempts,
        chunks_indexed=record.chunks_indexed,
        size_bytes=record.size_bytes,
        checksum=record.checksum,
        content_type=record.content_type,
        storage_uri=record.storage_uri,
        error=record.error if settings.debug_mode else ("" if not record.error else "Indexing failed"),
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _owns_document(request: Request, record) -> bool:
    """Whether the caller may see *record*.

    When auth is on, a document is visible to the key that uploaded it.
    Mirrors how conversation sessions are scoped — without it, any
    authenticated caller could enumerate every other tenant's filenames.
    """
    caller = getattr(request.state, "api_key_id", "")
    if not caller:
        return True  # auth disabled: single-tenant dev mode
    return not record.uploaded_by or record.uploaded_by == caller


@app.get(
    "/documents/{document_id}",
    response_model=DocumentStatusResponse,
    responses={404: {"model": ErrorResponse}},
    dependencies=[Depends(verify_api_key)],
)
async def document_status_endpoint(request: Request, document_id: str):
    """Current lifecycle state of an uploaded document."""
    from src.ingestion.registry import get_registry

    record = await asyncio.to_thread(get_registry().get, document_id)
    # 404 rather than 403 for someone else's document: a distinguishable
    # response would confirm the id exists.
    if record is None or not _owns_document(request, record):
        raise HTTPException(status_code=404, detail=f"Unknown document: {document_id}")
    return _to_status_response(record)


@app.post(
    "/documents/{document_id}/retry",
    response_model=DocumentRetryResponse,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(settings.heavy_rate_limit)
async def retry_document_endpoint(request: Request, response: Response, document_id: str):
    """Requeue a failed or dead-lettered document for indexing.

    The stored bytes are reused, so this recovers a document without the
    original file being uploaded again.
    """
    from src.ingestion.registry import get_registry
    from src.ingestion.replay import (
        OUTCOME_ALREADY_PROCESSED,
        OUTCOME_MISSING_OBJECT,
        OUTCOME_UNKNOWN_DOCUMENT,
        replay_document,
    )

    record = await asyncio.to_thread(get_registry().get, document_id)
    if record is None or not _owns_document(request, record):
        raise HTTPException(status_code=404, detail=f"Unknown document: {document_id}")

    result = await asyncio.to_thread(replay_document, document_id)

    if result.outcome == OUTCOME_UNKNOWN_DOCUMENT:
        raise HTTPException(status_code=404, detail=f"Unknown document: {document_id}")
    if result.outcome == OUTCOME_MISSING_OBJECT:
        # The bytes are gone, so no amount of retrying will help.
        raise HTTPException(status_code=409, detail=result.detail)
    if result.outcome == OUTCOME_ALREADY_PROCESSED:
        response.status_code = 200
    elif result.requeued:
        response.status_code = 202
    else:
        raise HTTPException(status_code=409, detail=result.detail)

    return DocumentRetryResponse(
        document_id=result.document_id,
        filename=result.filename,
        outcome=result.outcome,
        detail=result.detail,
        requeued=result.requeued,
        status_url=f"/documents/{document_id}",
    )


@app.get(
    "/documents",
    response_model=DocumentListResponse,
    dependencies=[Depends(verify_api_key)],
)
async def list_documents_endpoint(
    request: Request,
    status: str | None = None,
    department: str | None = None,
    limit: int = 50,
):
    """List registered documents, most recent first, with per-status counts."""
    from src.ingestion.registry import VALID_STATUSES, get_registry

    if status and status.upper() not in VALID_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status '{status}'. Allowed: {', '.join(sorted(VALID_STATUSES))}",
        )

    registry = get_registry()
    records = await asyncio.to_thread(
        registry.list_documents, status, department, max(1, min(limit, 500))
    )
    visible = [r for r in records if _owns_document(request, r)]
    stats = await asyncio.to_thread(registry.stats)
    return DocumentListResponse(
        documents=[_to_status_response(r) for r in visible],
        count=len(visible),
        stats=stats,
    )


# ---------------------------------------------------------------------------
# POST /eval
# ---------------------------------------------------------------------------

@app.get("/tools", dependencies=[Depends(verify_api_key)])
async def tools_endpoint():
    """List all available tools in the MCP registry."""
    from src.mcp.tool_registry import get_tool_registry

    registry = get_tool_registry()
    tools = registry.list_tools()
    return {
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
                "source": t.source,
            }
            for t in tools
        ],
        "count": len(tools),
    }


@app.post("/eval", response_model=EvalResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit(settings.heavy_rate_limit)
async def eval_endpoint(request: Request, response: Response, body: EvalRequest):
    """Run the RAGAS evaluation suite. This is a long-running operation."""
    # The suite is the most expensive thing this service does: every item
    # generates an answer, then four LLM-judged metrics score it. A daily
    # cap that /ask honours and this endpoint ignores is not a cap.
    #
    # Checked before the try block — HTTPException is an Exception, and the
    # handler below would otherwise turn this refusal into a 500 — and
    # before any work starts, since refusing after the spend saves nothing.
    if daily_cap_exceeded():
        raise HTTPException(
            status_code=503,
            detail="Daily cost cap reached. Evaluation resumes at the next UTC day.",
            headers={"Retry-After": str(_seconds_until_utc_midnight())},
        )

    try:
        from src.eval.ragas_eval import evaluate, load_eval_set
        from src.retrieval import get_retriever

        eval_set = load_eval_set(limit=body.limit)
        retriever = get_retriever(strategy=body.retriever_strategy)

        if body.mode == "graph":
            from src.graph.build_graph import ask as graph_ask
            answer_fn = lambda q: graph_ask(q, retriever_strategy=body.retriever_strategy)
        else:
            from src.rag.naive_rag import answer as naive_answer
            answer_fn = lambda q: naive_answer(q, retriever_strategy=body.retriever_strategy)

        start = time.time()
        scores = evaluate(answer_fn, retriever, eval_set=eval_set)
        duration = time.time() - start

        return EvalResponse(
            scores={k: round(v, 4) if isinstance(v, float) else v for k, v in scores.items()},
            items_evaluated=len(eval_set),
            mode=body.mode,
            retriever_strategy=body.retriever_strategy,
            duration_s=round(duration, 1),
        )
    except Exception as e:
        logger.exception("Eval endpoint failed")
        raise HTTPException(status_code=500, detail=_safe_error_detail(e)) from e
