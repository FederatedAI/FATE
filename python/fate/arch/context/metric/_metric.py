from ._type import Metric


class ScalarMetric(Metric):
    type = "scalar"

    def __init__(self, scalar) -> None:
        self.scalar = scalar

    def dict(self):
        return self.scalar


class LossMetric(Metric):
    type = "loss"

    def __init__(self, loss) -> None:
        self.loss = loss

    def dict(self) -> dict:
        return self.loss


class AccuracyMetric(Metric):
    type = "accuracy"

    def __init__(self, accuracy) -> None:
        self.accuracy = accuracy

    def dict(self) -> dict:
        return self.accuracy


class AUCMetric(Metric):
    type = "auc"

    def __init__(self, auc) -> None:
        self.auc = auc

    def dict(self) -> dict:
        return self.auc
