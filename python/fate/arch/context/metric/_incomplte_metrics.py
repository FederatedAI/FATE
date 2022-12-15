from ._type import InCompleteMetrics


class StepMetrics(InCompleteMetrics):
    complete = False

    def __init__(self, name, type, data) -> None:
        self.name = name
        self.type = type
        self.data = data

    @classmethod
    def from_step_metric(cls, name, metric, step, timestamp):
        return StepMetrics(name, metric.type, [dict(metric=metric.dict(), step=step, timestamp=timestamp)])

    def merge(self, metrics: InCompleteMetrics):
        if not isinstance(metrics, StepMetrics):
            raise ValueError(f"can't merge metrics type `{metrics}` with StepMetrics")
        if metrics.type != self.type:
            raise ValueError(f"can't merge metrics type `{metrics}` with StepMetrics named `{self.name}`")
        return StepMetrics(name=self.name, type=self.type, data=[*self.data, *metrics.data])

    def dict(self) -> dict:
        return dict(
            name=self.name,
            type=self.type,
            data=self.data,
        )

    @classmethod
    def from_dict(cls, d) -> "StepMetrics":
        return StepMetrics(**d)
