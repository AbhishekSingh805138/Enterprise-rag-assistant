"""Streamlit chat UI for the Enterprise RAG Assistant.

Connects to the FastAPI backend for queries. Supports streaming responses
with citation highlighting.

Usage:
    streamlit run ui/app.py
    # or with custom API URL:
    API_URL=http://localhost:8000 streamlit run ui/app.py
"""
from __future__ import annotations

import json
import os
import uuid

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
# API key for authenticating against the backend when AUTH_ENABLED=true.
RAG_API_KEY = os.getenv("RAG_API_KEY", "")


def _auth_headers() -> dict[str, str]:
    """Authorization header for API requests (empty when auth is not configured)."""
    if RAG_API_KEY:
        return {"Authorization": f"Bearer {RAG_API_KEY}"}
    return {}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* ---- Global ---- */
.block-container { padding-top: 2rem; }

/* ---- Sidebar polish ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1724 0%, #1a2332 100%);
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown span,
section[data-testid="stSidebar"] .stMarkdown label,
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.08) !important;
}

/* ---- Status badge ---- */
.status-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 14px; border-radius: 20px; font-size: 0.82rem;
    font-weight: 600; margin: 4px 0 8px;
}
.status-ok   { background: #064e3b; color: #6ee7b7; }
.status-err  { background: #7f1d1d; color: #fca5a5; }

/* ---- Metadata pills ---- */
.meta-row {
    display: flex; flex-wrap: wrap; gap: 6px;
    margin-top: 8px; padding-top: 8px;
    border-top: 1px solid rgba(128,128,128,0.15);
}
.meta-pill {
    display: inline-flex; align-items: center; gap: 4px;
    background: rgba(99,102,241,0.10); color: #a5b4fc;
    padding: 3px 10px; border-radius: 12px;
    font-size: 0.74rem; font-weight: 500;
    white-space: nowrap;
}
.meta-pill.cache-hit {
    background: rgba(16,185,129,0.15); color: #6ee7b7;
}

/* ---- Pipeline progress ---- */
.pipeline-bar {
    display: flex; flex-wrap: wrap; gap: 4px;
    margin: 6px 0 4px;
}
.node-chip {
    padding: 2px 8px; border-radius: 8px;
    font-size: 0.7rem; font-weight: 500;
    background: rgba(99,102,241,0.12); color: #818cf8;
    transition: all 0.2s;
}
.node-chip.active {
    background: rgba(99,102,241,0.3); color: #c7d2fe;
    box-shadow: 0 0 6px rgba(99,102,241,0.3);
}

/* ---- Welcome card ---- */
.welcome-card {
    text-align: center; padding: 3rem 2rem;
    border: 1px solid rgba(128,128,128,0.15);
    border-radius: 16px; margin: 2rem auto;
    max-width: 640px;
    background: rgba(15,23,42,0.3);
}
.welcome-card h2 { margin-bottom: 0.5rem; }
.welcome-card p { color: #94a3b8; font-size: 0.95rem; }

.suggestion-grid {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 10px; margin-top: 1.5rem; text-align: left;
}
.suggestion-btn {
    padding: 12px 16px; border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.2);
    background: rgba(30,41,59,0.5);
    color: #cbd5e1; font-size: 0.85rem;
    cursor: pointer; transition: all 0.15s;
}
.suggestion-btn:hover {
    border-color: rgba(99,102,241,0.5);
    background: rgba(99,102,241,0.08);
}
.suggestion-btn .label { color: #94a3b8; font-size: 0.72rem; margin-bottom: 4px; }

/* ---- Sidebar info cards ---- */
.info-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px; padding: 10px 14px;
    margin: 6px 0;
}
.info-card .title {
    font-size: 0.72rem; color: #64748b;
    text-transform: uppercase; letter-spacing: 0.5px;
    margin-bottom: 3px;
}
.info-card .value {
    font-size: 0.95rem; color: #e2e8f0; font-weight: 600;
}
.info-card .sub { font-size: 0.75rem; color: #64748b; margin-top: 2px; }

/* ---- Footer ---- */
.sidebar-footer {
    text-align: center; padding: 10px 0;
    font-size: 0.72rem; color: #475569;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# Helper: render metadata pills
# ---------------------------------------------------------------------------

_NODE_LABELS = {
    "guardrail_check": "Guard",
    "load_memory": "Memory",
    "scope_check": "Scope",
    "intent_detect": "Intent",
    "query_transform": "Transform",
    "query_transform_node": "Transform",
    "planner": "Plan",
    "retrieve": "Retrieve",
    "grade_documents": "Grade",
    "transform_query": "Rewrite",
    "web_search": "Web",
    "generate": "Generate",
    "critic": "Critic",
    "cache_store": "Cache",
    "save_memory": "Save",
    "process_sub_queries_parallel": "Sub-Qs",
    "synthesize": "Synthesize",
}


def _render_meta_pills(meta: dict) -> str:
    """Build HTML for metadata pills row."""
    pills = []
    if meta.get("mode"):
        pills.append(f'<span class="meta-pill">{meta["mode"]}</span>')
    if meta.get("retriever"):
        pills.append(f'<span class="meta-pill">{meta["retriever"]}</span>')
    if meta.get("tokens"):
        pills.append(f'<span class="meta-pill">{meta["tokens"]} tokens</span>')
    if meta.get("cost") is not None:
        pills.append(f'<span class="meta-pill">${meta["cost"]:.5f}</span>')
    if meta.get("latency"):
        secs = meta["latency"] / 1000
        pills.append(f'<span class="meta-pill">{secs:.1f}s</span>')
    if meta.get("intent"):
        pills.append(f'<span class="meta-pill">{meta["intent"]}</span>')
    if meta.get("cache_hit"):
        pills.append('<span class="meta-pill cache-hit">CACHE HIT</span>')
    return f'<div class="meta-row">{"".join(pills)}</div>' if pills else ""


def _render_node_progress(nodes: list[str], active: str | None = None) -> str:
    """Build HTML for pipeline node progress bar."""
    chips = []
    for n in nodes:
        label = _NODE_LABELS.get(n, n.replace("_", " ").title())
        cls = "node-chip active" if n == active else "node-chip"
        chips.append(f'<span class="{cls}">{label}</span>')
    return f'<div class="pipeline-bar">{"".join(chips)}</div>'


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### :office: Enterprise RAG")

    st.markdown("**Pipeline**")
    mode = st.selectbox(
        "Mode",
        ["graph", "naive", "auto"],
        index=0,
        help="**Graph**: Full CRAG pipeline with grading, critic, memory. "
             "**Naive**: Fast single-shot RAG chain. "
             "**Auto**: Picks based on query complexity.",
        label_visibility="collapsed",
    )

    retriever = st.selectbox(
        "Retriever",
        ["hybrid", "dense", "multi_query", "rerank", "cross_rerank",
         "hybrid_cross_rerank", "knowledge_graph"],
        index=0,
        help="Retrieval strategy for document search.",
        label_visibility="collapsed",
    )

    st.divider()

    # Health check
    _api_ok = False
    try:
        _health_resp = requests.get(f"{API_URL}/health", timeout=3)
        if _health_resp.ok:
            _health = _health_resp.json()
            _api_ok = True
            st.markdown(
                f'<div class="status-badge status-ok">Connected</div>',
                unsafe_allow_html=True,
            )
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f'<div class="info-card">'
                    f'<div class="title">Collection</div>'
                    f'<div class="value">{_health["document_count"]}</div>'
                    f'<div class="sub">chunks indexed</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    f'<div class="info-card">'
                    f'<div class="title">Version</div>'
                    f'<div class="value">{_health["version"]}</div>'
                    f'<div class="sub">{_health["collection"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="status-badge status-err">Unhealthy</div>',
                unsafe_allow_html=True,
            )
    except requests.ConnectionError:
        st.markdown(
            '<div class="status-badge status-err">Disconnected</div>',
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.warning(f"Health check failed: {e}")

    st.divider()

    # Document upload
    st.markdown("**Upload Documents**")
    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "txt", "md"],
        help="Supported: PDF, TXT, Markdown",
        label_visibility="collapsed",
    )
    department = st.selectbox(
        "Department",
        ["general", "hr", "legal", "engineering", "finance", "security", "operations"],
        help="Tag the document with a department for filtered retrieval.",
    )

    if uploaded_file is not None and st.button("Upload & Ingest", use_container_width=True, type="primary"):
        with st.spinner(f"Ingesting {uploaded_file.name}..."):
            try:
                resp = requests.post(
                    f"{API_URL}/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "application/octet-stream")},
                    data={"department": department},
                    headers=_auth_headers(),
                    timeout=60,
                )
                if resp.ok:
                    result = resp.json()
                    st.success(
                        f"**{result['filename']}** ingested  \n"
                        f"{result['chunks_added']} new chunks "
                        f"({result['collection_total']} total)"
                    )
                else:
                    st.error(f"Upload failed: {resp.json().get('detail', resp.text)}")
            except requests.ConnectionError:
                st.error(f"Cannot connect to API at {API_URL}")
            except Exception as e:
                st.error(f"Upload error: {e}")

    st.divider()

    # Session management
    st.markdown("**Conversation**")
    st.markdown(
        f'<div class="info-card">'
        f'<div class="title">Session ID</div>'
        f'<div class="value" style="font-family:monospace;font-size:0.85rem">{st.session_state.session_id}</div>'
        f'<div class="sub">{len(st.session_state.messages) // 2} exchange(s)</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if st.button("New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = uuid.uuid4().hex[:12]
        st.rerun()

    st.divider()

    st.markdown(
        '<div class="sidebar-footer">'
        'Enterprise RAG Assistant v1.0<br>'
        'LangChain + LangGraph + ChromaDB'
        '</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

# Welcome screen when no messages
if not st.session_state.messages:
    st.markdown(
        '<div class="welcome-card">'
        '<h2>Enterprise RAG Assistant</h2>'
        '<p>Ask questions about company policies, procedures, and documents.<br>'
        'Answers are grounded in your enterprise knowledge base with source citations.</p>'
        '<div class="suggestion-grid">'
        '<div class="suggestion-btn"><div class="label">HR</div>What is the PTO and leave policy?</div>'
        '<div class="suggestion-btn"><div class="label">Engineering</div>How does the incident response process work?</div>'
        '<div class="suggestion-btn"><div class="label">Legal</div>What are the data protection requirements?</div>'
        '<div class="suggestion-btn"><div class="label">Finance</div>What is the procurement approval process?</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🏢" if msg["role"] == "assistant" else None):
        st.markdown(msg["content"])
        if msg.get("meta"):
            st.markdown(_render_meta_pills(msg["meta"]), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Ask a question about enterprise documents..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream response from API
    with st.chat_message("assistant", avatar="🏢"):
        placeholder = st.empty()
        progress_placeholder = st.empty()
        meta_placeholder = st.empty()

        if not _api_ok:
            placeholder.error(
                f"Cannot connect to API at {API_URL}.  \n"
                "Start the server: `uvicorn api.app:app`"
            )
        else:
            try:
                resp = requests.post(
                    f"{API_URL}/ask",
                    json={
                        "question": prompt,
                        "mode": mode,
                        "retriever_strategy": retriever,
                        "stream": True,
                        "session_id": st.session_state.session_id,
                    },
                    headers=_auth_headers(),
                    stream=True,
                    timeout=120,
                )
                resp.raise_for_status()

                full_text = ""
                meta = {}
                visited_nodes: list[str] = []
                current_node = None

                for line in resp.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    data = json.loads(line[6:])

                    if data.get("type") == "token":
                        content = data.get("content", "")
                        if mode == "graph" or (mode == "auto" and data.get("node")):
                            # Graph mode sends full generation at once
                            full_text = content
                        else:
                            # Naive mode streams token by token
                            full_text += content
                        placeholder.markdown(full_text + " **|**")

                    elif data.get("type") == "status":
                        node = data.get("node", "")
                        if node and node not in visited_nodes:
                            visited_nodes.append(node)
                        current_node = node
                        progress_placeholder.markdown(
                            _render_node_progress(visited_nodes, current_node),
                            unsafe_allow_html=True,
                        )

                    elif data.get("type") == "done":
                        full_text = data.get("answer", full_text)
                        meta = {
                            "mode": mode,
                            "retriever": retriever,
                            "cost": data.get("cost_usd", 0),
                            "latency": data.get("latency_ms", 0),
                            "tokens": data.get("tokens_used", 0),
                            "intent": data.get("intent"),
                            "cache_hit": data.get("cache_hit", False),
                        }

                    elif data.get("type") == "error":
                        err_msg = data.get("message", "An unknown error occurred.")
                        placeholder.error(f"Pipeline error: {err_msg}")

                # Final display
                placeholder.markdown(full_text)
                progress_placeholder.empty()
                if meta:
                    meta_placeholder.markdown(
                        _render_meta_pills(meta), unsafe_allow_html=True
                    )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_text,
                    "meta": meta,
                })

            except requests.ConnectionError:
                placeholder.error(
                    f"Cannot connect to API at {API_URL}.  \n"
                    "Start the API server with: `uvicorn api.app:app`"
                )
            except requests.HTTPError as e:
                detail = e.response.text
                try:
                    detail = e.response.json().get("detail", detail)
                except Exception:
                    pass
                placeholder.error(f"API error ({e.response.status_code}): {detail}")
            except Exception as e:
                placeholder.error(f"Error: {e}")
