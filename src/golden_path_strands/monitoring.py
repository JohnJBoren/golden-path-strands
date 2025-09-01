"""Monitoring helpers integrating with OpenTelemetry."""
from __future__ import annotations

from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource


def setup_telemetry(service_name: str, endpoint: str) -> None:
    """Configure OpenTelemetry metrics exporter."""
    resource = Resource.create({"service.name": service_name})
    exporter = OTLPMetricExporter(endpoint=endpoint)
    reader = PeriodicExportingMetricReader(exporter)
    provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(provider)


class MetricsCollector:
    """Simple wrapper around OpenTelemetry metrics."""

    def __init__(self) -> None:
        self._meter = metrics.get_meter(__name__)
        self._exploration_counter = self._meter.create_counter("exploration_count")
        self._evaluation_latency = self._meter.create_histogram("evaluation_latency_ms")

    def record_exploration(self) -> None:
        self._exploration_counter.add(1)

    def record_evaluation_latency(self, latency_ms: float) -> None:
        self._evaluation_latency.record(latency_ms)
