"""OpenTelemetry spans, including across the ingestion queue.

The request id already correlates log lines across the API, the queue and
the worker. What it cannot do is show where the time went: an upload that
took four minutes to become queryable is three spans in two processes,
and only a trace shows which one was slow.

The queue boundary is the part worth getting right. A trace normally
propagates over HTTP headers; here the carrier is the event envelope, so
the context is injected into the event payload on publish and extracted
by the worker as a *link* plus parent — which is how a trace survives a
hop that may happen minutes later in a different process.

Entirely optional. With ``OTEL_ENABLED=false`` (the default) every
function here is a no-op that costs a boolean check, and the SDK is never
imported — so the dependency stays optional for anyone who does not run a
collector.
"""
from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from typing import Any

from config import settings

logger = logging.getLogger(__name__)

# Key under which the propagation context travels inside an event payload.
TRACE_CONTEXT_KEY = "_trace_context"

_tracer: Any = None
_initialised = False
_available = False


def is_enabled() -> bool:
    return bool(settings.otel_enabled)


def setup_tracing(service_name: str | None = None) -> bool:
    """Initialise the tracer provider. Returns True when tracing is live.

    Safe to call more than once and from any entry point (API, worker,
    scripts) — later calls are no-ops.
    """
    global _tracer, _initialised, _available
    if _initialised:
        return _available
    _initialised = True

    if not is_enabled():
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as e:
        # Configured but not installed. Loud, because a deployment that
        # believes it has traces and does not will debug the wrong thing.
        logger.error(
            "OTEL_ENABLED=true but the OpenTelemetry SDK is not installed (%s). "
            "Install opentelemetry-sdk and opentelemetry-exporter-otlp, or set "
            "OTEL_ENABLED=false. Running without traces.", e,
        )
        return False

    try:
        resource = Resource.create({
            "service.name": service_name or settings.otel_service_name,
            "service.version": settings.otel_service_version,
        })
        provider = TracerProvider(resource=resource)
        if settings.otel_exporter_endpoint:
            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otel_exporter_endpoint))
            )
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer(__name__)
        _available = True
        logger.info(
            "OpenTelemetry tracing enabled (service=%s exporter=%s)",
            resource.attributes.get("service.name"),
            settings.otel_exporter_endpoint or "none (spans are dropped)",
        )
    except Exception as e:
        logger.error("Could not initialise OpenTelemetry: %s. Running without traces.", e)
        _available = False
    return _available


@contextlib.contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """Start a span, or do nothing when tracing is off.

    Never raises on account of tracing: an exporter problem must not be
    able to fail the operation being traced.
    """
    if not setup_tracing():
        yield None
        return
    try:
        with _tracer.start_as_current_span(name) as current:
            for key, value in attributes.items():
                if value is not None:
                    current.set_attribute(key, value)
            yield current
    except Exception:
        logger.debug("Span %r failed; continuing untraced", name, exc_info=True)
        yield None


def inject_context(payload: dict) -> dict:
    """Embed the current trace context into an event payload.

    Returns *payload* unchanged when tracing is off, so the envelope of a
    non-traced deployment gains no extra field.
    """
    if not setup_tracing():
        return payload
    try:
        from opentelemetry.propagate import inject

        carrier: dict[str, str] = {}
        inject(carrier)
        if carrier:
            payload = {**payload, TRACE_CONTEXT_KEY: carrier}
    except Exception:
        logger.debug("Could not inject trace context", exc_info=True)
    return payload


def extract_context(payload: dict) -> Any:
    """Recover the publishing context from an event payload, or None."""
    if not setup_tracing():
        return None
    carrier = (payload or {}).get(TRACE_CONTEXT_KEY)
    if not isinstance(carrier, dict):
        return None
    try:
        from opentelemetry.propagate import extract

        return extract(carrier)
    except Exception:
        logger.debug("Could not extract trace context", exc_info=True)
        return None


@contextlib.contextmanager
def consumer_span(name: str, payload: dict, **attributes: Any) -> Iterator[Any]:
    """Span for work resumed from a queued event.

    Parented to the publishing span when the envelope carries a context,
    which is what joins the upload request and the indexing that happens
    minutes later into one trace.
    """
    context = extract_context(payload)
    if context is None:
        with span(name, **attributes) as current:
            yield current
        return
    try:
        with _tracer.start_as_current_span(name, context=context) as current:
            for key, value in attributes.items():
                if value is not None:
                    current.set_attribute(key, value)
            yield current
    except Exception:
        logger.debug("Consumer span %r failed; continuing untraced", name, exc_info=True)
        yield None


def reset_for_tests() -> None:
    global _tracer, _initialised, _available
    _tracer, _initialised, _available = None, False, False
