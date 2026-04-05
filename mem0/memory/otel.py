"""OpenTelemetry tracing and metrics for mem0.

Auto-enabled when ``opentelemetry-api`` is importable and the
``MEM0_OTEL_ENABLED`` environment variable is **not** set to ``"false"``.

All public helpers are **safe no-ops** when OTel is unavailable or disabled,
so callers never need to guard imports.
"""

import functools
import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Auto-detection & opt-out
# ---------------------------------------------------------------------------

_OTEL_AVAILABLE = False
_OTEL_ENABLED = False

try:
    from opentelemetry import baggage, context, trace
    from opentelemetry.metrics import get_meter
    from opentelemetry.trace import SpanKind, StatusCode

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover
    pass

if _OTEL_AVAILABLE:
    _env_flag = os.environ.get("MEM0_OTEL_ENABLED", "").strip().lower()
    # Enabled by default when OTel is installed; opt-out with "false" / "0" / "no"
    _OTEL_ENABLED = _env_flag not in ("false", "0", "no")


def is_otel_enabled() -> bool:
    """Return whether OpenTelemetry instrumentation is active."""
    return _OTEL_ENABLED


# ---------------------------------------------------------------------------
# Startup diagnostics
# ---------------------------------------------------------------------------

# Well-known OTel / Azure Monitor env vars to surface in the diagnostic log.
_OTEL_ENV_VARS = (
    "OTEL_SERVICE_NAME",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
    "OTEL_EXPORTER_OTLP_HEADERS",
    "OTEL_TRACES_EXPORTER",
    "OTEL_METRICS_EXPORTER",
    "APPLICATIONINSIGHTS_CONNECTION_STRING",
)

_diagnostics_logged = False


def log_otel_diagnostics() -> None:
    """Log a one-time startup summary of OpenTelemetry readiness.

    Emits an INFO-level message describing:
    * whether the ``opentelemetry`` package is installed,
    * whether instrumentation is enabled or opted-out,
    * which OTel / Azure Monitor environment variables are set,
    * whether a real ``TracerProvider`` (not the default no-op) is active.

    Safe to call multiple times — only logs once per process.
    """
    global _diagnostics_logged
    if _diagnostics_logged:
        return
    _diagnostics_logged = True

    if not _OTEL_AVAILABLE:
        logger.info(
            "mem0 OpenTelemetry: DISABLED — 'opentelemetry-api' package is not installed. "
            "Install with: pip install \"mem0ai[telemetry]\" (or pip install opentelemetry-api opentelemetry-sdk)"
        )
        return

    if not _OTEL_ENABLED:
        logger.info(
            "mem0 OpenTelemetry: DISABLED — opted out via MEM0_OTEL_ENABLED=%s",
            os.environ.get("MEM0_OTEL_ENABLED", ""),
        )
        return

    # OTel is available & enabled — report env vars and provider status.
    env_present = {var: os.environ[var] for var in _OTEL_ENV_VARS if var in os.environ}
    env_missing = [var for var in _OTEL_ENV_VARS if var not in os.environ]

    # Check whether a real (non-default) TracerProvider has been configured.
    has_real_provider = False
    provider_name = "unknown"
    try:
        tp = trace.get_tracer_provider()
        provider_name = type(tp).__name__
        # The SDK's ProxyTracerProvider wraps the real one — check for that too.
        has_real_provider = provider_name not in ("NoOpTracerProvider", "ProxyTracerProvider")
    except Exception:
        pass

    # Build log lines
    lines = ["mem0 OpenTelemetry: ENABLED"]

    if env_present:
        # Mask sensitive values (connection strings, headers, etc.)
        safe_vars = []
        for k, v in env_present.items():
            if any(s in k.upper() for s in ("KEY", "SECRET", "TOKEN", "PASSWORD", "CONNECTION_STRING", "HEADERS")):
                safe_vars.append(f"{k}=<set>")
            else:
                safe_vars.append(f"{k}={v}")
        lines.append(f"  Environment variables: {', '.join(safe_vars)}")
    else:
        lines.append(
            "  Environment variables: none of the standard OTEL_*/APPLICATIONINSIGHTS_* vars are set."
        )

    if has_real_provider:
        lines.append(f"  TracerProvider: {provider_name} (OK — traces will be exported)")
    else:
        lines.append(
            f"  TracerProvider: {provider_name} (no SDK provider configured yet — "
            "traces will be collected once the host process sets up a TracerProvider, "
            "e.g. via opentelemetry-sdk or azure-monitor-opentelemetry-exporter)"
        )

    # Actionable hints for common misconfigurations
    hints = []
    if "OTEL_EXPORTER_OTLP_ENDPOINT" not in os.environ and "APPLICATIONINSIGHTS_CONNECTION_STRING" not in os.environ:
        hints.append(
            "Hint: Set OTEL_EXPORTER_OTLP_ENDPOINT or APPLICATIONINSIGHTS_CONNECTION_STRING "
            "so the host process knows where to send telemetry."
        )
    if "OTEL_SERVICE_NAME" not in os.environ:
        hints.append("Hint: Set OTEL_SERVICE_NAME to identify this service in your telemetry backend.")
    for hint in hints:
        lines.append(f"  {hint}")

    logger.info("\n".join(lines))


# ---------------------------------------------------------------------------
# Tracer & Meter singletons (no-ops when disabled)
# ---------------------------------------------------------------------------

_tracer = None  # type: Any
_meter = None  # type: Any
_op_counter = None  # type: Any
_op_duration = None  # type: Any
_op_errors = None  # type: Any


def _init_otel():
    """Lazily initialise tracer & meter once the package version is known."""
    global _tracer, _meter, _op_counter, _op_duration, _op_errors

    if _tracer is not None:
        return  # already initialised

    try:
        import mem0

        version = mem0.__version__
    except Exception:
        version = "0.0.0"

    _tracer = trace.get_tracer("mem0", version)
    _meter = get_meter("mem0", version)

    _op_counter = _meter.create_counter(
        "mem0.operation.count",
        description="Number of mem0 operations",
    )
    _op_duration = _meter.create_histogram(
        "mem0.operation.duration",
        unit="ms",
        description="Duration of mem0 operations in milliseconds",
    )
    _op_errors = _meter.create_counter(
        "mem0.operation.errors",
        description="Number of failed mem0 operations",
    )


def _get_tracer():
    if not _OTEL_ENABLED:
        return None
    _init_otel()
    return _tracer


# ---------------------------------------------------------------------------
# Baggage helpers
# ---------------------------------------------------------------------------

_BAGGAGE_KEYS = ("coworker.correlationId", "coworker.id")


def _attach_baggage_attributes(span) -> None:
    """Read well-known baggage entries and set them as span attributes."""
    ctx = context.get_current()
    for key in _BAGGAGE_KEYS:
        value = baggage.get_baggage(key, ctx)
        if value:
            span.set_attribute(key, value)


# ---------------------------------------------------------------------------
# Attribute extraction
# ---------------------------------------------------------------------------

_KNOWN_KWARGS = {
    "user_id": "mem0.user_id",
    "agent_id": "mem0.agent_id",
    "run_id": "mem0.run_id",
    "memory_id": "mem0.memory_id",
    "infer": "mem0.infer",
}


def _set_common_attributes(span, kwargs: Dict[str, Any], provider_attrs: Optional[Dict[str, str]] = None) -> None:
    """Set standard mem0 span attributes from method kwargs."""
    for kwarg_name, attr_name in _KNOWN_KWARGS.items():
        value = kwargs.get(kwarg_name)
        if value is not None:
            span.set_attribute(attr_name, value)
    # Alias: emit mem0.thread_id = run_id for coworker KQL compatibility
    run_id = kwargs.get("run_id")
    if run_id is not None:
        span.set_attribute("mem0.thread_id", run_id)
    # Static provider-level attrs (set once at __init__)
    if provider_attrs:
        for attr_name, attr_value in provider_attrs.items():
            span.set_attribute(attr_name, attr_value)


# ---------------------------------------------------------------------------
# Decorators: @traced / @async_traced
# ---------------------------------------------------------------------------

def traced(operation_name: str) -> Callable:
    """Decorator that wraps a **sync** method in an OTel span.

    The span is named ``mem0.{operation_name}`` with ``SpanKind.INTERNAL``.
    If OTel is disabled the original function is returned unchanged.
    """

    def decorator(fn: Callable) -> Callable:
        if not _OTEL_ENABLED:
            return fn

        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            tracer = _get_tracer()
            if tracer is None:
                return fn(self, *args, **kwargs)

            span_name = f"mem0.{operation_name}"
            provider_attrs = getattr(self, "_otel_provider_attrs", None)
            with tracer.start_as_current_span(span_name, kind=SpanKind.INTERNAL) as span:
                span.set_attribute("mem0.operation", operation_name)
                _set_common_attributes(span, kwargs, provider_attrs)
                _attach_baggage_attributes(span)

                _op_counter.add(1, {"operation": operation_name})
                start = time.monotonic()
                try:
                    result = fn(self, *args, **kwargs)
                    _op_duration.record(
                        (time.monotonic() - start) * 1000.0,
                        {"operation": operation_name},
                    )
                    return result
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(StatusCode.ERROR, str(exc))
                    _op_errors.add(1, {"operation": operation_name, "error_type": type(exc).__name__})
                    _op_duration.record(
                        (time.monotonic() - start) * 1000.0,
                        {"operation": operation_name},
                    )
                    raise

        return wrapper

    return decorator


def async_traced(operation_name: str) -> Callable:
    """Decorator that wraps an **async** method in an OTel span.

    Same semantics as :func:`traced` but for ``async def`` methods.
    """

    def decorator(fn: Callable) -> Callable:
        if not _OTEL_ENABLED:
            return fn

        @functools.wraps(fn)
        async def wrapper(self, *args, **kwargs):
            tracer = _get_tracer()
            if tracer is None:
                return await fn(self, *args, **kwargs)

            span_name = f"mem0.{operation_name}"
            provider_attrs = getattr(self, "_otel_provider_attrs", None)
            with tracer.start_as_current_span(span_name, kind=SpanKind.INTERNAL) as span:
                span.set_attribute("mem0.operation", operation_name)
                _set_common_attributes(span, kwargs, provider_attrs)
                _attach_baggage_attributes(span)

                _op_counter.add(1, {"operation": operation_name})
                start = time.monotonic()
                try:
                    result = await fn(self, *args, **kwargs)
                    _op_duration.record(
                        (time.monotonic() - start) * 1000.0,
                        {"operation": operation_name},
                    )
                    return result
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(StatusCode.ERROR, str(exc))
                    _op_errors.add(1, {"operation": operation_name, "error_type": type(exc).__name__})
                    _op_duration.record(
                        (time.monotonic() - start) * 1000.0,
                        {"operation": operation_name},
                    )
                    raise

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Child-span context managers (for internal methods)
# ---------------------------------------------------------------------------

@contextmanager
def child_span(span_name: str, **attributes):
    """Open a child span and yield it.  No-op when OTel is disabled."""
    if not _OTEL_ENABLED:
        yield None
        return
    tracer = _get_tracer()
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(span_name, kind=SpanKind.INTERNAL) as span:
        for k, v in attributes.items():
            span.set_attribute(k, v)
        yield span


# ---------------------------------------------------------------------------
# Span event helpers
# ---------------------------------------------------------------------------

def record_memory_event(event_name: str, memory_id: Optional[str] = None, operation: Optional[str] = None) -> None:
    """Emit a span event on the **current** span (no-op when OTel is disabled).

    Event names follow the ``mem0.memory.{action}`` convention:
        - ``mem0.memory.created``
        - ``mem0.memory.updated``
        - ``mem0.memory.deleted``
        - ``mem0.memory.noop``

    Attributes attached: ``memory_id``, ``operation``.
    """
    if not _OTEL_ENABLED:
        return
    span = trace.get_current_span()
    if span is None or not span.is_recording():
        return
    attrs: Dict[str, Any] = {}
    if memory_id is not None:
        attrs["memory_id"] = memory_id
    if operation is not None:
        attrs["operation"] = operation
    span.add_event(event_name, attributes=attrs)


def set_search_result_attributes(result_count: int, has_graph: bool) -> None:
    """Set ``mem0.result.count`` and ``mem0.has_graph`` on the current span."""
    if not _OTEL_ENABLED:
        return
    span = trace.get_current_span()
    if span is None or not span.is_recording():
        return
    span.set_attribute("mem0.result.count", result_count)
    span.set_attribute("mem0.has_graph", has_graph)
