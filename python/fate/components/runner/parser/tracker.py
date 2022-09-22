import typing

from fate.interface import Metric, MetricMeta


class Tracker:
    def __init__(self, tracker) -> None:
        self._tracker = tracker

    @classmethod
    def parse(cls, tracker):
        return Tracker(tracker)

    def log_metric_data(
        self, metric_namespace: str, metric_name: str, metrics: typing.List[Metric]
    ):
        return self._tracker.log_metric_data(
            metric_namespace=metric_namespace,
            metric_name=metric_name,
            metrics=[
                dict(key=metric.key, value=metric.value, timestamp=metric.timestamp)
                for metric in metrics
            ],
        )

    def log_metric_meta(
        self, metric_namespace: str, metric_name: str, metric_meta: MetricMeta
    ):
        return self._tracker.set_metric_meta(
            metric_namespace=metric_namespace,
            metric_name=metric_name,
            metric_meta=dict(
                name=metric_meta.name,
                metric_type=metric_meta.metric_type,
                metas=metric_meta.metas,
                extra_metas=metric_meta.extra_metas,
            ),
        )

    def log_component_summary(self, summary_data: dict):
        return self._tracker.log_component_summary(summary_data=summary_data)
