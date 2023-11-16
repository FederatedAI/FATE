import os
import functools
from opentelemetry import trace

_ENABLE_TRACING = None
_ENABLE_TRACING_DEFAULT = True


def _is_tracing_enabled():
    global _ENABLE_TRACING
    if _ENABLE_TRACING is None:
        _ENABLE_TRACING = os.environ.get("FATE_ENABLE_TRACING", str(_ENABLE_TRACING_DEFAULT)).lower() == "true"
    return _ENABLE_TRACING


def setup_tracing(service_name, endpoint: str = None):
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider

    from opentelemetry.sdk.trace.export import BatchSpanProcessor as SpanProcessor

    # from opentelemetry.sdk.trace.export import SimpleSpanProcessor as SpanProcessor

    provider = TracerProvider(resource=Resource(attributes={SERVICE_NAME: service_name}))
    provider.add_span_processor(SpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)


def auto_trace(func=None, *, annotation=None):
    if annotation is not None:

        def _auto_trace(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return _trace_func(func, args, kwargs, span_name=annotation)

            return wrapper

        return _auto_trace

    else:

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return _trace_func(func, args, kwargs)

        return wrapper


def _trace_func(func, args, kwargs, span_name=None):
    if not _is_tracing_enabled():
        return func(*args, **kwargs)

    module_name = func.__module__
    qualname = func.__qualname__
    if span_name is None:
        span_name = qualname
    tracer = get_tracer(module_name)
    with tracer.start_as_current_span(span_name) as span:
        import traceback

        callstack = "\n".join([line.strip() for line in traceback.format_stack()[:-2]])
        span.set_attribute("call_stack", callstack)
        span.set_attribute("qualname", qualname)
        span.set_attribute("module", module_name)
        return func(*args, **kwargs)


def inject_carrier():
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    carrier = {}
    TraceContextTextMapPropagator().inject(carrier)
    return carrier


def extract_carrier(carrier):
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    return TraceContextTextMapPropagator().extract(carrier)


def federation_auto_trace(func):
    if not _is_tracing_enabled():
        return func

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        name = kwargs.get("name")
        tag = kwargs.get("tag")

        module_name = func.__module__
        qualname = func.__qualname__
        span_name = f"federation.{qualname}"
        tracer = get_tracer(module_name)
        with tracer.start_as_current_span(span_name) as span:
            import traceback

            callstack = "\n".join([line.strip() for line in traceback.format_stack()[:-2]])
            span.set_attribute("call_stack", callstack)
            span.set_attribute("qualname", qualname)
            span.set_attribute("module", module_name)
            span.set_attribute("name", name)
            span.set_attribute("tag", tag)

            return func(*args, **kwargs)

    return wrapper


class WrappedTracer(trace.Tracer):
    def __init__(self, tracer):
        self._tracer = tracer

    def start_as_current_span(self, *args, **kwargs):
        return self._tracer.start_as_current_span(*args, **kwargs)

    def start_span(self, *args, **kwargs):
        return self._tracer.start_span(*args, **kwargs)


def get_tracer(module_name):
    if not _is_tracing_enabled():
        return trace.NoOpTracer()
    return WrappedTracer(trace.get_tracer(module_name))


__all__ = ["setup_tracing", "auto_trace", "inject_carrier", "extract_carrier", "get_tracer", "federation_auto_trace"]
