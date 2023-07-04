import pydantic

from ._fields import StringChoice


class LRSchedulerType(StringChoice):
    choice = {'constant', 'linear', 'step'}


class LRSchedulerParam(pydantic.BaseModel):
    method: LRSchedulerType = 'constant'
    scheduler_params: dict = None


def lr_scheduler_param():
    namespace = {}
    return type("LRSchedulerParam", (LRSchedulerParam,), namespace)
