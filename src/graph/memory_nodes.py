"""Graph nodes for loading and saving conversation memory.

These nodes bookend the CRAG graph: load_memory runs at the start to
inject conversation history into state, and save_memory runs after the
critic to persist the Q/A pair for future turns.
"""
from __future__ import annotations

import logging

from config import settings
from src.graph.tracing import traced

logger = logging.getLogger(__name__)


@traced
def load_memory(state: dict) -> dict:
    """Load conversation history for the current session into state."""
    session_id = state.get("session_id", "")
    if not session_id or not settings.memory_enabled:
        return {"chat_history": [], "memory_context": ""}

    try:
        from src.memory.context_builder import build_memory_context
        from src.memory.conversation_store import get_conversation_store

        store = get_conversation_store()
        history = store.get_history(session_id)
        memory_context = build_memory_context(history)

        logger.info(
            "Loaded %d history messages for session %s (context: %d chars)",
            len(history), session_id[:12], len(memory_context),
        )
        return {
            "chat_history": history,
            "memory_context": memory_context,
        }
    except Exception:
        logger.debug("Failed to load memory — continuing without history", exc_info=True)
        return {"chat_history": [], "memory_context": ""}


@traced
def save_memory(state: dict) -> dict:
    """Persist the current Q/A pair to conversation history."""
    session_id = state.get("session_id", "")
    if not session_id or not settings.memory_enabled:
        return {}

    question = state.get("original_question", state.get("question", ""))
    generation = state.get("generation", "")

    if not question or not generation:
        return {}

    try:
        from src.memory.conversation_store import get_conversation_store

        store = get_conversation_store()
        store.add_message(session_id, "user", question)
        store.add_message(session_id, "assistant", generation)
        logger.info("Saved Q/A to session %s", session_id[:12])
    except Exception:
        logger.debug("Failed to save memory", exc_info=True)

    return {}
