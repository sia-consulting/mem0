"""Tests for mem0 OpenTelemetry instrumentation (mem0/memory/otel.py)."""

import importlib
import logging
import os
import sys
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers: OTel in-memory exporter fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def otel_setup():
    """Set up OTel with in-memory span exporter and metric reader for test assertions."""
    from opentelemetry import trace
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    # Force-set the global tracer provider for each test
    trace._TRACER_PROVIDER = None
    trace._TRACER_PROVIDER_SET_ONCE._done = False
    trace.set_tracer_provider(tracer_provider)

    metric_reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[metric_reader])

    # Reset the otel module so it picks up the new providers
    import mem0.memory.otel as otel_mod

    otel_mod._OTEL_AVAILABLE = True
    otel_mod._OTEL_ENABLED = True
    otel_mod._tracer = None  # force re-init
    otel_mod._meter = None
    otel_mod._init_otel()

    # Patch the meter to use our MeterProvider
    otel_mod._meter = meter_provider.get_meter("mem0", "test")
    otel_mod._op_counter = otel_mod._meter.create_counter(
        "mem0.operation.count", description="Number of mem0 operations"
    )
    otel_mod._op_duration = otel_mod._meter.create_histogram(
        "mem0.operation.duration", unit="ms", description="Duration of mem0 operations in milliseconds"
    )
    otel_mod._op_errors = otel_mod._meter.create_counter(
        "mem0.operation.errors", description="Number of failed mem0 operations"
    )

    yield {
        "span_exporter": span_exporter,
        "metric_reader": metric_reader,
        "tracer_provider": tracer_provider,
        "meter_provider": meter_provider,
    }

    # Cleanup
    tracer_provider.shutdown()
    meter_provider.shutdown()


# ---------------------------------------------------------------------------
# Tests: Auto-enable / disable logic
# ---------------------------------------------------------------------------


class TestOtelEnableDisable:
    """Verify auto-enable and opt-out via MEM0_OTEL_ENABLED."""

    def test_noop_when_otel_not_installed(self):
        """Decorators should be pass-through when OTel is not installed."""
        import mem0.memory.otel as otel_mod

        orig_available = otel_mod._OTEL_AVAILABLE
        orig_enabled = otel_mod._OTEL_ENABLED
        try:
            otel_mod._OTEL_AVAILABLE = False
            otel_mod._OTEL_ENABLED = False

            # traced should return the function unchanged
            @otel_mod.traced("test_op")
            def my_func(self):
                return 42

            assert my_func(None) == 42
        finally:
            otel_mod._OTEL_AVAILABLE = orig_available
            otel_mod._OTEL_ENABLED = orig_enabled

    def test_noop_when_disabled_via_env(self):
        """When MEM0_OTEL_ENABLED=false, decorators should be pass-through."""
        import mem0.memory.otel as otel_mod

        orig_enabled = otel_mod._OTEL_ENABLED
        try:
            otel_mod._OTEL_ENABLED = False

            @otel_mod.traced("test_op")
            def my_func(self):
                return "hello"

            assert my_func(None) == "hello"
        finally:
            otel_mod._OTEL_ENABLED = orig_enabled

    def test_is_otel_enabled_returns_current_state(self):
        """is_otel_enabled() should reflect the module state."""
        import mem0.memory.otel as otel_mod

        orig = otel_mod._OTEL_ENABLED
        try:
            otel_mod._OTEL_ENABLED = True
            assert otel_mod.is_otel_enabled() is True
            otel_mod._OTEL_ENABLED = False
            assert otel_mod.is_otel_enabled() is False
        finally:
            otel_mod._OTEL_ENABLED = orig


# ---------------------------------------------------------------------------
# Tests: Span creation and attributes
# ---------------------------------------------------------------------------


class TestSpanCreation:
    """Verify spans are created with correct names, kind, and attributes."""

    def test_traced_creates_span_with_correct_name(self, otel_setup):
        """@traced should create a span named mem0.{operation}."""
        import mem0.memory.otel as otel_mod
        from opentelemetry.trace import SpanKind

        # Need to re-create decorator after otel_setup modifies module state
        @otel_mod.traced("test_search")
        def fake_search(self, query, *, user_id=None, run_id=None):
            return {"results": []}

        fake_self = MagicMock()
        fake_self._otel_provider_attrs = {"mem0.collection": "test_col"}
        result = fake_search(fake_self, "hello", user_id="u1", run_id="r1")

        assert result == {"results": []}

        spans = otel_setup["span_exporter"].get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "mem0.test_search"
        assert span.kind == SpanKind.INTERNAL
        assert span.attributes["mem0.operation"] == "test_search"
        assert span.attributes["mem0.user_id"] == "u1"
        assert span.attributes["mem0.run_id"] == "r1"
        assert span.attributes["mem0.thread_id"] == "r1"  # alias
        assert span.attributes["mem0.collection"] == "test_col"

    def test_traced_records_agent_id(self, otel_setup):
        """@traced should set mem0.agent_id attribute."""
        import mem0.memory.otel as otel_mod

        @otel_mod.traced("test_add")
        def fake_add(self, messages, *, agent_id=None):
            return {"results": []}

        fake_self = MagicMock()
        fake_self._otel_provider_attrs = None
        fake_add(fake_self, "msg", agent_id="agent_007")

        spans = otel_setup["span_exporter"].get_finished_spans()
        assert spans[0].attributes["mem0.agent_id"] == "agent_007"

    def test_traced_records_memory_id(self, otel_setup):
        """@traced should set mem0.memory_id attribute."""
        import mem0.memory.otel as otel_mod

        @otel_mod.traced("test_get")
        def fake_get(self, *, memory_id=None):
            return {}

        fake_self = MagicMock()
        fake_self._otel_provider_attrs = None
        fake_get(fake_self, memory_id="mem_123")

        spans = otel_setup["span_exporter"].get_finished_spans()
        assert spans[0].attributes["mem0.memory_id"] == "mem_123"

    def test_traced_records_infer_attribute(self, otel_setup):
        """@traced should set mem0.infer attribute."""
        import mem0.memory.otel as otel_mod

        @otel_mod.traced("test_add")
        def fake_add(self, messages, *, infer=True):
            return {}

        fake_self = MagicMock()
        fake_self._otel_provider_attrs = None
        fake_add(fake_self, "msg", infer=False)

        spans = otel_setup["span_exporter"].get_finished_spans()
        assert spans[0].attributes["mem0.infer"] is False


# ---------------------------------------------------------------------------
# Tests: Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Verify exception recording on spans."""

    def test_traced_records_exception(self, otel_setup):
        """@traced should record exceptions and set ERROR status."""
        import mem0.memory.otel as otel_mod
        from opentelemetry.trace import StatusCode

        @otel_mod.traced("test_fail")
        def failing_method(self):
            raise ValueError("test error")

        fake_self = MagicMock()
        fake_self._otel_provider_attrs = None

        with pytest.raises(ValueError, match="test error"):
            failing_method(fake_self)

        spans = otel_setup["span_exporter"].get_finished_spans()
        assert len(spans) == 1
        assert spans[0].status.status_code == StatusCode.ERROR
        assert "test error" in spans[0].status.description

        # Should have recorded exception event
        events = spans[0].events
        assert any(e.name == "exception" for e in events)


# ---------------------------------------------------------------------------
# Tests: Async traced
# ---------------------------------------------------------------------------


class TestAsyncTraced:
    """Verify async tracing works correctly."""

    @pytest.mark.asyncio
    async def test_async_traced_creates_span(self, otel_setup):
        """@async_traced should create spans for async methods."""
        import mem0.memory.otel as otel_mod

        @otel_mod.async_traced("async_search")
        async def fake_async_search(self, query, *, user_id=None):
            return {"results": []}

        fake_self = MagicMock()
        fake_self._otel_provider_attrs = {"mem0.llm.provider": "openai"}
        result = await fake_async_search(fake_self, "hello", user_id="u2")

        assert result == {"results": []}
        spans = otel_setup["span_exporter"].get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "mem0.async_search"
        assert spans[0].attributes["mem0.user_id"] == "u2"
        assert spans[0].attributes["mem0.llm.provider"] == "openai"

    @pytest.mark.asyncio
    async def test_async_traced_records_exception(self, otel_setup):
        """@async_traced should record exceptions and set ERROR status."""
        import mem0.memory.otel as otel_mod
        from opentelemetry.trace import StatusCode

        @otel_mod.async_traced("async_fail")
        async def failing_async(self):
            raise RuntimeError("async fail")

        fake_self = MagicMock()
        fake_self._otel_provider_attrs = None

        with pytest.raises(RuntimeError, match="async fail"):
            await failing_async(fake_self)

        spans = otel_setup["span_exporter"].get_finished_spans()
        assert spans[0].status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# Tests: Baggage propagation
# ---------------------------------------------------------------------------


class TestBaggagePropagation:
    """Verify coworker.correlationId and coworker.id are read from baggage."""

    def test_correlation_id_from_baggage(self, otel_setup):
        """coworker.correlationId should be set as span attribute when present in baggage."""
        import mem0.memory.otel as otel_mod
        from opentelemetry import baggage, context

        @otel_mod.traced("test_baggage")
        def fake_op(self):
            return "ok"

        fake_self = MagicMock()
        fake_self._otel_provider_attrs = None

        # Set baggage in context
        ctx = baggage.set_baggage("coworker.correlationId", "corr-123")
        ctx = baggage.set_baggage("coworker.id", "cw-456", context=ctx)
        token = context.attach(ctx)
        try:
            fake_op(fake_self)
        finally:
            context.detach(token)

        spans = otel_setup["span_exporter"].get_finished_spans()
        assert spans[0].attributes["coworker.correlationId"] == "corr-123"
        assert spans[0].attributes["coworker.id"] == "cw-456"

    def test_no_baggage_no_attributes(self, otel_setup):
        """When no baggage is set, coworker attributes should not appear."""
        import mem0.memory.otel as otel_mod

        @otel_mod.traced("test_no_baggage")
        def fake_op(self):
            return "ok"

        fake_self = MagicMock()
        fake_self._otel_provider_attrs = None
        fake_op(fake_self)

        spans = otel_setup["span_exporter"].get_finished_spans()
        assert "coworker.correlationId" not in spans[0].attributes
        assert "coworker.id" not in spans[0].attributes


# ---------------------------------------------------------------------------
# Tests: Span events
# ---------------------------------------------------------------------------


class TestSpanEvents:
    """Verify span events for memory operations."""

    def test_record_memory_event_created(self, otel_setup):
        """record_memory_event should add a span event."""
        import mem0.memory.otel as otel_mod
        from opentelemetry import trace

        tracer = otel_mod._get_tracer()
        with tracer.start_as_current_span("test_parent") as span:
            otel_mod.record_memory_event("mem0.memory.created", memory_id="m1", operation="add")

        spans = otel_setup["span_exporter"].get_finished_spans()
        parent_span = [s for s in spans if s.name == "test_parent"][0]
        events = parent_span.events
        assert len(events) == 1
        assert events[0].name == "mem0.memory.created"
        assert events[0].attributes["memory_id"] == "m1"
        assert events[0].attributes["operation"] == "add"

    def test_record_memory_event_noop(self, otel_setup):
        """record_memory_event should work for noop events."""
        import mem0.memory.otel as otel_mod

        tracer = otel_mod._get_tracer()
        with tracer.start_as_current_span("test_noop") as span:
            otel_mod.record_memory_event("mem0.memory.noop", memory_id="m2", operation="noop")

        spans = otel_setup["span_exporter"].get_finished_spans()
        parent_span = [s for s in spans if s.name == "test_noop"][0]
        assert parent_span.events[0].name == "mem0.memory.noop"

    def test_record_memory_event_noop_when_disabled(self):
        """record_memory_event should be a no-op when OTel is disabled."""
        import mem0.memory.otel as otel_mod

        orig = otel_mod._OTEL_ENABLED
        try:
            otel_mod._OTEL_ENABLED = False
            # Should not raise
            otel_mod.record_memory_event("mem0.memory.created", memory_id="m1", operation="add")
        finally:
            otel_mod._OTEL_ENABLED = orig


# ---------------------------------------------------------------------------
# Tests: Search result attributes
# ---------------------------------------------------------------------------


class TestSearchResultAttributes:
    """Verify mem0.result.count and mem0.has_graph on search spans."""

    def test_set_search_result_attributes(self, otel_setup):
        """set_search_result_attributes should set result count and graph flag."""
        import mem0.memory.otel as otel_mod

        tracer = otel_mod._get_tracer()
        with tracer.start_as_current_span("test_search_attrs"):
            otel_mod.set_search_result_attributes(result_count=5, has_graph=True)

        spans = otel_setup["span_exporter"].get_finished_spans()
        span = [s for s in spans if s.name == "test_search_attrs"][0]
        assert span.attributes["mem0.result.count"] == 5
        assert span.attributes["mem0.has_graph"] is True

    def test_set_search_result_no_graph(self, otel_setup):
        """set_search_result_attributes with has_graph=False."""
        import mem0.memory.otel as otel_mod

        tracer = otel_mod._get_tracer()
        with tracer.start_as_current_span("test_no_graph"):
            otel_mod.set_search_result_attributes(result_count=0, has_graph=False)

        spans = otel_setup["span_exporter"].get_finished_spans()
        span = [s for s in spans if s.name == "test_no_graph"][0]
        assert span.attributes["mem0.result.count"] == 0
        assert span.attributes["mem0.has_graph"] is False


# ---------------------------------------------------------------------------
# Tests: Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    """Verify meter instruments are updated by traced decorators."""

    def test_operation_count_incremented(self, otel_setup):
        """Counter mem0.operation.count should be incremented on traced call."""
        import mem0.memory.otel as otel_mod

        @otel_mod.traced("metrics_test")
        def fake_op(self):
            return "ok"

        fake_self = MagicMock()
        fake_self._otel_provider_attrs = None
        fake_op(fake_self)
        fake_op(fake_self)

        metrics_data = otel_setup["metric_reader"].get_metrics_data()
        # Find the counter
        found_count = False
        for resource_metric in metrics_data.resource_metrics:
            for scope_metric in resource_metric.scope_metrics:
                for metric in scope_metric.metrics:
                    if metric.name == "mem0.operation.count":
                        for dp in metric.data.data_points:
                            if dp.attributes.get("operation") == "metrics_test":
                                assert dp.value == 2
                                found_count = True
        assert found_count, "mem0.operation.count metric not found"

    def test_operation_duration_recorded(self, otel_setup):
        """Histogram mem0.operation.duration should be recorded."""
        import mem0.memory.otel as otel_mod

        @otel_mod.traced("duration_test")
        def fake_op(self):
            return "ok"

        fake_self = MagicMock()
        fake_self._otel_provider_attrs = None
        fake_op(fake_self)

        metrics_data = otel_setup["metric_reader"].get_metrics_data()
        found_duration = False
        for resource_metric in metrics_data.resource_metrics:
            for scope_metric in resource_metric.scope_metrics:
                for metric in scope_metric.metrics:
                    if metric.name == "mem0.operation.duration":
                        for dp in metric.data.data_points:
                            if dp.attributes.get("operation") == "duration_test":
                                assert dp.count == 1
                                assert dp.sum >= 0  # duration should be non-negative
                                found_duration = True
        assert found_duration, "mem0.operation.duration metric not found"

    def test_error_counter_incremented_on_failure(self, otel_setup):
        """Counter mem0.operation.errors should be incremented on exception."""
        import mem0.memory.otel as otel_mod

        @otel_mod.traced("error_test")
        def failing_op(self):
            raise TypeError("bad type")

        fake_self = MagicMock()
        fake_self._otel_provider_attrs = None

        with pytest.raises(TypeError):
            failing_op(fake_self)

        metrics_data = otel_setup["metric_reader"].get_metrics_data()
        found_errors = False
        for resource_metric in metrics_data.resource_metrics:
            for scope_metric in resource_metric.scope_metrics:
                for metric in scope_metric.metrics:
                    if metric.name == "mem0.operation.errors":
                        for dp in metric.data.data_points:
                            if dp.attributes.get("operation") == "error_test":
                                assert dp.value == 1
                                assert dp.attributes.get("error_type") == "TypeError"
                                found_errors = True
        assert found_errors, "mem0.operation.errors metric not found"


# ---------------------------------------------------------------------------
# Tests: Child span helper
# ---------------------------------------------------------------------------


class TestChildSpan:
    """Verify child_span context manager."""

    def test_child_span_creates_span(self, otel_setup):
        """child_span should create a nested span."""
        import mem0.memory.otel as otel_mod

        tracer = otel_mod._get_tracer()
        with tracer.start_as_current_span("parent"):
            with otel_mod.child_span("mem0.vector_store.search", provider="qdrant") as cs:
                assert cs is not None

        spans = otel_setup["span_exporter"].get_finished_spans()
        names = {s.name for s in spans}
        assert "parent" in names
        assert "mem0.vector_store.search" in names

        child = [s for s in spans if s.name == "mem0.vector_store.search"][0]
        assert child.attributes["provider"] == "qdrant"

    def test_child_span_noop_when_disabled(self):
        """child_span should yield None when OTel is disabled."""
        import mem0.memory.otel as otel_mod

        orig = otel_mod._OTEL_ENABLED
        try:
            otel_mod._OTEL_ENABLED = False
            with otel_mod.child_span("test_noop") as cs:
                assert cs is None
        finally:
            otel_mod._OTEL_ENABLED = orig


# ---------------------------------------------------------------------------
# Tests: Tracer name
# ---------------------------------------------------------------------------


class TestTracerName:
    """Verify the tracer is named 'mem0'."""

    def test_tracer_name_is_mem0(self, otel_setup):
        """The tracer instrumentation scope should be 'mem0'."""
        import mem0.memory.otel as otel_mod

        @otel_mod.traced("name_test")
        def fake_op(self):
            return "ok"

        fake_self = MagicMock()
        fake_self._otel_provider_attrs = None
        fake_op(fake_self)

        spans = otel_setup["span_exporter"].get_finished_spans()
        assert spans[0].instrumentation_scope.name == "mem0"


# ---------------------------------------------------------------------------
# Tests: Startup diagnostics (log_otel_diagnostics)
# ---------------------------------------------------------------------------


class TestOtelDiagnostics:
    """Verify log_otel_diagnostics() output for various configurations."""

    def _reset_diagnostics(self):
        """Reset the one-time guard so diagnostics can be re-tested."""
        import mem0.memory.otel as otel_mod
        otel_mod._diagnostics_logged = False

    def test_logs_disabled_when_otel_not_installed(self, caplog):
        """Should log 'DISABLED' with install hint when OTel is not available."""
        import mem0.memory.otel as otel_mod

        orig_avail = otel_mod._OTEL_AVAILABLE
        orig_enabled = otel_mod._OTEL_ENABLED
        self._reset_diagnostics()
        try:
            otel_mod._OTEL_AVAILABLE = False
            otel_mod._OTEL_ENABLED = False

            with caplog.at_level(logging.INFO, logger="mem0.memory.otel"):
                otel_mod.log_otel_diagnostics()

            assert "DISABLED" in caplog.text
            assert "not installed" in caplog.text
            assert "pip install" in caplog.text
        finally:
            otel_mod._OTEL_AVAILABLE = orig_avail
            otel_mod._OTEL_ENABLED = orig_enabled
            self._reset_diagnostics()

    def test_logs_disabled_when_opted_out(self, caplog):
        """Should log 'DISABLED' with opted-out message."""
        import mem0.memory.otel as otel_mod

        orig_avail = otel_mod._OTEL_AVAILABLE
        orig_enabled = otel_mod._OTEL_ENABLED
        self._reset_diagnostics()
        try:
            otel_mod._OTEL_AVAILABLE = True
            otel_mod._OTEL_ENABLED = False

            with patch.dict(os.environ, {"MEM0_OTEL_ENABLED": "false"}):
                with caplog.at_level(logging.INFO, logger="mem0.memory.otel"):
                    otel_mod.log_otel_diagnostics()

            assert "DISABLED" in caplog.text
            assert "opted out" in caplog.text
        finally:
            otel_mod._OTEL_AVAILABLE = orig_avail
            otel_mod._OTEL_ENABLED = orig_enabled
            self._reset_diagnostics()

    def test_logs_enabled_with_env_vars(self, caplog, otel_setup):
        """Should log 'ENABLED' and list detected env vars."""
        import mem0.memory.otel as otel_mod

        self._reset_diagnostics()
        env = {
            "OTEL_SERVICE_NAME": "my-service",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4317",
        }
        try:
            with patch.dict(os.environ, env, clear=False):
                with caplog.at_level(logging.INFO, logger="mem0.memory.otel"):
                    otel_mod.log_otel_diagnostics()

            assert "ENABLED" in caplog.text
            assert "OTEL_SERVICE_NAME=my-service" in caplog.text
            assert "OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317" in caplog.text
        finally:
            self._reset_diagnostics()

    def test_masks_sensitive_env_vars(self, caplog, otel_setup):
        """Sensitive env vars (connection strings, headers) should be masked."""
        import mem0.memory.otel as otel_mod

        self._reset_diagnostics()
        env = {
            "APPLICATIONINSIGHTS_CONNECTION_STRING": "InstrumentationKey=secret123",
            "OTEL_EXPORTER_OTLP_HEADERS": "api-key=supersecret",
        }
        try:
            with patch.dict(os.environ, env, clear=False):
                with caplog.at_level(logging.INFO, logger="mem0.memory.otel"):
                    otel_mod.log_otel_diagnostics()

            assert "ENABLED" in caplog.text
            assert "APPLICATIONINSIGHTS_CONNECTION_STRING=<set>" in caplog.text
            assert "OTEL_EXPORTER_OTLP_HEADERS=<set>" in caplog.text
            assert "secret123" not in caplog.text
            assert "supersecret" not in caplog.text
        finally:
            self._reset_diagnostics()

    def test_logs_hints_when_no_endpoint(self, caplog, otel_setup):
        """Should show hints when no exporter endpoint is configured."""
        import mem0.memory.otel as otel_mod

        self._reset_diagnostics()
        # Clear all OTel env vars to trigger hints
        remove_vars = {var: "" for var in otel_mod._OTEL_ENV_VARS}
        try:
            with patch.dict(os.environ, {}, clear=False):
                # Remove any OTel vars that might be set
                for var in otel_mod._OTEL_ENV_VARS:
                    os.environ.pop(var, None)
                with caplog.at_level(logging.INFO, logger="mem0.memory.otel"):
                    otel_mod.log_otel_diagnostics()

            assert "ENABLED" in caplog.text
            assert "Hint:" in caplog.text
            assert "OTEL_EXPORTER_OTLP_ENDPOINT" in caplog.text
            assert "OTEL_SERVICE_NAME" in caplog.text
        finally:
            self._reset_diagnostics()

    def test_only_logs_once(self, caplog, otel_setup):
        """log_otel_diagnostics should only emit output once per process."""
        import mem0.memory.otel as otel_mod

        self._reset_diagnostics()
        try:
            with caplog.at_level(logging.INFO, logger="mem0.memory.otel"):
                otel_mod.log_otel_diagnostics()
                first_output = caplog.text
                caplog.clear()
                otel_mod.log_otel_diagnostics()
                second_output = caplog.text

            assert "ENABLED" in first_output or "DISABLED" in first_output
            # Second call should produce no output
            assert second_output == ""
        finally:
            self._reset_diagnostics()
