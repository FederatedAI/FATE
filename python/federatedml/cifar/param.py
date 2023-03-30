from federatedml.param.base_param import BaseParam


class CifarParam(BaseParam):
    def __init__(self, config: dict = {}):
        super(CifarParam, self).__init__()
        self.config = config

    def check(self):
        ...
