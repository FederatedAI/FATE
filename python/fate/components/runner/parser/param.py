from fate.interface import Params as ParamInterface


class Params(ParamInterface):
    def __init__(self):
        pass

    def get(self, names, default):
        cur = self
        for name in names.split("."):
            try:
                getattr(cur, name)
            except AttributeError:
                return default
        return cur

    def update(self, data):
        return self

    @property
    def is_need_run(self) -> bool:
        return self.get("need_run", True)

    @property
    def is_need_cv(self) -> bool:
        return self.get("cv_param.need_cv", False)

    @property
    def is_need_stepwise(self) -> bool:
        return self.get("stepwise_param.need_stepwise", False)
