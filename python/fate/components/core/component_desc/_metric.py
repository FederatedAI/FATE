from fate.arch.context._metrics import (
    BaseMetricsHandler,
    InMemoryMetricsHandler,
    OneTimeMetrics,
    StepMetrics,
)

from .artifacts.metric import JsonMetricFileWriter, JsonMetricRestfulWriter


class ComponentMetricsFileHandler(InMemoryMetricsHandler):
    def __init__(self, writer: JsonMetricFileWriter) -> None:
        self._writer = writer
        super().__init__()

    def finalize(self):
        self._writer.write(self.get_metrics())


class ComponentMetricsRestfulHandler(BaseMetricsHandler):
    def __init__(self, writer: JsonMetricRestfulWriter) -> None:
        self._writer = writer

    def _log_step_metrics(self, metrics: "StepMetrics"):
        record = metrics.to_record()
        self._writer.write(record.dict())

    def _log_one_time_metrics(self, metrics: "OneTimeMetrics"):
        record = metrics.to_record()
        self._writer.write(record.dict())

    def finalize(self):
        self._writer.close()
