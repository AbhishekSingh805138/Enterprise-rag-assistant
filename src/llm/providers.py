"""Chat and embedding model construction, decoupled from one vendor.

``ChatOpenAI`` and ``OpenAIEmbeddings`` were instantiated directly in nine
modules. An OpenAI outage was therefore a total outage: the circuit
breakers failed fast, but there was nothing to fail over *to*, and
because embeddings were equally coupled even a cached, already-indexed
corpus became unqueryable.

Two asymmetric problems, handled differently:

**Chat models are interchangeable.** A second vendor can answer the same
prompt, so a fallback is genuinely useful and is wired in
``build_chat_model`` + ``ChatModelWithFallbacks``.

**Embedding models are not.** Vectors from two models occupy different
spaces; querying a corpus indexed by one using another returns confident
nonsense with no error anywhere. So there is deliberately *no* embedding
fallback — instead ``embedding_fingerprint`` lets the vector store refuse
to serve an index built by a different model, which turns a silent
retrieval-quality collapse into a startup failure.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableConfig

from config import settings

logger = logging.getLogger(__name__)

OPENAI = "openai"
AZURE_OPENAI = "azure_openai"
ANTHROPIC = "anthropic"

CHAT_PROVIDERS = (OPENAI, AZURE_OPENAI, ANTHROPIC)
# Anthropic has no embedding API, so it cannot back the vector store.
EMBEDDING_PROVIDERS = (OPENAI, AZURE_OPENAI)


class ProviderNotAvailable(RuntimeError):
    """A provider was configured but its package or credentials are missing."""


# ---------------------------------------------------------------------------
# Chat models
# ---------------------------------------------------------------------------

def build_chat_model(
    provider: str,
    model: str,
    temperature: float = 0.0,
) -> BaseChatModel:
    """Construct a chat model for *provider*.

    Imports are deferred so an uninstalled optional vendor package only
    fails when that vendor is actually configured — installing the
    Anthropic SDK should not be a prerequisite for running on OpenAI.
    """
    provider = provider.strip().lower()
    common = {
        "temperature": temperature,
        "timeout": settings.llm_timeout,
        "max_retries": settings.llm_max_retries,
    }

    if provider == OPENAI:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, api_key=settings.openai_api_key, **common)

    if provider == AZURE_OPENAI:
        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError as e:  # pragma: no cover - langchain_openai is required
            raise ProviderNotAvailable(f"azure_openai requires langchain-openai: {e}") from e
        if not settings.azure_openai_endpoint:
            raise ProviderNotAvailable(
                "LLM_PROVIDER=azure_openai requires AZURE_OPENAI_ENDPOINT."
            )
        return AzureChatOpenAI(
            azure_deployment=settings.azure_openai_deployment or model,
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key or settings.openai_api_key,
            api_version=settings.azure_openai_api_version,
            **common,
        )

    if provider == ANTHROPIC:
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:
            raise ProviderNotAvailable(
                "LLM_PROVIDER=anthropic requires the langchain-anthropic package "
                "(pip install langchain-anthropic)."
            ) from e
        if not settings.anthropic_api_key:
            raise ProviderNotAvailable(
                "LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY."
            )
        # ChatAnthropic names the request deadline differently.
        return ChatAnthropic(
            model=model,
            api_key=settings.anthropic_api_key,
            temperature=temperature,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
        )

    raise ProviderNotAvailable(
        f"Unknown LLM provider {provider!r}. Choose from: {', '.join(CHAT_PROVIDERS)}"
    )


class ChatModelWithFallbacks(Runnable):
    """A chat model that fails over to another vendor.

    Deliberately *not* a bare ``primary.with_fallbacks([...])``. That
    returns a ``RunnableWithFallbacks``, which has no
    ``with_structured_output`` — and this codebase builds typed graders,
    planners and critics through exactly that method in a dozen places.
    Returning one would have broken every structured node the moment a
    fallback was configured, which is the worst possible time to find out.

    So structured output is composed per-provider and *then* given
    fallbacks, keeping the typed contract on both sides.
    """

    def __init__(self, primary: BaseChatModel, fallbacks: Sequence[BaseChatModel]):
        self.primary = primary
        self.fallbacks = list(fallbacks)
        self._chain = primary.with_fallbacks(self.fallbacks) if self.fallbacks else primary

    # -- Runnable ---------------------------------------------------------
    def invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        return self._chain.invoke(input, config, **kwargs)

    async def ainvoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        return await self._chain.ainvoke(input, config, **kwargs)

    def stream(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any):
        return self._chain.stream(input, config, **kwargs)

    async def astream(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any):
        async for chunk in self._chain.astream(input, config, **kwargs):
            yield chunk

    def batch(self, inputs: list, config: RunnableConfig | None = None, **kwargs: Any) -> list:
        return self._chain.batch(inputs, config, **kwargs)

    # -- Chat-model surface used across the graph -------------------------
    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable:
        """Typed output on the primary, with typed fallbacks behind it."""
        primary = self.primary.with_structured_output(schema, **kwargs)
        if not self.fallbacks:
            return primary
        alternatives = []
        for model in self.fallbacks:
            try:
                alternatives.append(model.with_structured_output(schema, **kwargs))
            except Exception:
                # A vendor that cannot do structured output for this schema
                # is skipped rather than breaking the primary path.
                logger.warning(
                    "Fallback %s cannot produce structured output for %s; "
                    "it will not cover this call",
                    type(model).__name__,
                    getattr(schema, "__name__", schema),
                )
        return primary.with_fallbacks(alternatives) if alternatives else primary

    def bind_tools(self, tools: Any, **kwargs: Any) -> Runnable:
        primary = self.primary.bind_tools(tools, **kwargs)
        alternatives = [m.bind_tools(tools, **kwargs) for m in self.fallbacks]
        return primary.with_fallbacks(alternatives) if alternatives else primary

    def __getattr__(self, name: str) -> Any:
        # Attribute reads (model_name, temperature, ...) resolve against the
        # primary, so callers inspecting the model see the configured one.
        return getattr(self.__dict__["primary"], name)

    def __repr__(self) -> str:
        names = ", ".join(type(m).__name__ for m in self.fallbacks) or "none"
        return f"ChatModelWithFallbacks(primary={type(self.primary).__name__}, fallbacks=[{names}])"


def fallback_spec() -> tuple[str, str] | None:
    """The configured fallback ``(provider, model)``, or None if disabled."""
    provider = settings.llm_fallback_provider.strip().lower()
    if not provider:
        return None
    model = settings.llm_fallback_model.strip() or settings.llm_model
    return provider, model


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def build_embeddings(provider: str, model: str) -> Embeddings:
    """Construct an embedding model for *provider*."""
    provider = provider.strip().lower()

    if provider == OPENAI:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model, api_key=settings.openai_api_key)

    if provider == AZURE_OPENAI:
        from langchain_openai import AzureOpenAIEmbeddings

        if not settings.azure_openai_endpoint:
            raise ProviderNotAvailable(
                "EMBEDDING_PROVIDER=azure_openai requires AZURE_OPENAI_ENDPOINT."
            )
        return AzureOpenAIEmbeddings(
            azure_deployment=settings.azure_embedding_deployment or model,
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key or settings.openai_api_key,
            api_version=settings.azure_openai_api_version,
        )

    if provider == ANTHROPIC:
        raise ProviderNotAvailable(
            "anthropic has no embedding API; EMBEDDING_PROVIDER must be one of: "
            + ", ".join(EMBEDDING_PROVIDERS)
        )

    raise ProviderNotAvailable(
        f"Unknown embedding provider {provider!r}. "
        f"Choose from: {', '.join(EMBEDDING_PROVIDERS)}"
    )


def embedding_fingerprint() -> str:
    """Identity of the embedding space the current settings produce.

    Stored alongside the collection so a changed embedding model is
    caught. Azure deployments are named freely, so the deployment name is
    part of the identity — two deployments can serve different models
    under the same ``EMBEDDING_MODEL`` value.
    """
    provider = settings.embedding_provider.strip().lower()
    parts = [provider, settings.embedding_model]
    if provider == AZURE_OPENAI:
        parts.append(settings.azure_embedding_deployment or "")
    return "|".join(parts)
