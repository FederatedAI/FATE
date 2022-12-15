from ._type import InCompleteMetrics


class StepMetrics(InCompleteMetrics):
    complete = False

    def __init__(self, name, type, data, namespace, groups, metadata) -> None:
        self.name = name
        self.type = type
        self.namespace = namespace
        self.groups = groups
        self.data = data
        self.metadata = metadata

    def merge(self, metrics: InCompleteMetrics):
        if not isinstance(metrics, StepMetrics):
            raise ValueError(f"can't merge metrics type `{metrics}` with StepMetrics")
        if metrics.type != self.type or metrics.nemaspace != self.namespace:
            raise ValueError(f"can't merge metrics type `{metrics}` with StepMetrics named `{self.name}`")
        # TODO: compare groups
        return StepMetrics(
            name=self.name,
            type=self.type,
            namespace=self.namespace,
            groups=self.groups,
            data=[*self.data, *metrics.data],
            metadata=self.metadata,
        )

    def dict(self) -> dict:
        return dict(
            name=self.name,
            namespace=self.nemaspace,
            groups=self.groups,
            type=self.type,
            metadata=self.metadata,
            data=self.data,
        )

    @classmethod
    def from_dict(cls, d) -> "StepMetrics":
        return StepMetrics(**d)
