import pydantic


class Parameter:
    def parse(self, obj):
        raise NotImplementedError()

    def dict(self):
        raise NotImplementedError()


class ConInt(Parameter):
    def __init__(self, gt: int = None, ge: int = None, lt: int = None, le: int = None) -> None:
        self.gt = gt
        self.ge = ge
        self.lt = lt
        self.le = le

    def parse(self, obj):
        return pydantic.parse_obj_as(pydantic.conint(gt=self.gt, ge=self.ge, lt=self.lt, le=self.le), obj)

    def dict(self):
        meta = {}
        if self.gt is not None:
            meta["gt"] = self.gt
        if self.ge is not None:
            meta["ge"] = self.ge
        if self.lt is not None:
            meta["lt"] = self.lt
        if self.le is not None:
            meta["le"] = self.le
        return meta


class ConFloat(Parameter):
    def __init__(self, gt: float = None, ge: float = None, lt: float = None, le: float = None) -> None:
        self.gt = gt
        self.ge = ge
        self.lt = lt
        self.le = le

    def parse(self, obj):
        return pydantic.parse_obj_as(pydantic.confloat(gt=self.gt, ge=self.ge, lt=self.lt, le=self.le), obj)

    def dict(self):
        meta = {}
        if self.gt is not None:
            meta["gt"] = self.gt
        if self.ge is not None:
            meta["ge"] = self.ge
        if self.lt is not None:
            meta["lt"] = self.lt
        if self.le is not None:
            meta["le"] = self.le
        return meta


def parse(parameter_type, obj):
    if isinstance(parameter_type, Parameter):
        return parameter_type.parse(obj)
    else:
        return pydantic.parse_obj_as(parameter_type, obj)
