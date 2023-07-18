import pydantic

from ._fields import conint


class CVParam(pydantic.BaseModel):
    n_splits: conint(gt=1)
    shuffle: bool = False
    random_state: int = None


def cv_param():
    namespace = {}
    return type("CVParam", (CVParam,), namespace)
