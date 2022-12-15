from ._type import Metrics


class ROCMetrics(Metrics):
    type = "roc"

    def __init__(self, name, data) -> None:
        self.name = name
        self.data = data

    def dict(self) -> dict:
        return dict(
            name=self.name,
            type=self.type,
            data=self.data,
        )
