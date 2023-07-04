from ._fields import ConstrainedFloat


class LearningRate(ConstrainedFloat):
    gt = 0.0

    @classmethod
    def dict(cls):
        return {"name": cls.__name__}


def learning_rate_param():
    return LearningRate
