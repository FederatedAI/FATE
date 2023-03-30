from federatedml.param.base_param import BaseParam


class CifarParam(BaseParam):
    def __init__(self, config: dict = {}, data_root=""):
        super(CifarParam, self).__init__()
        self.config = config
        self.data_root = data_root

    def check(self):
        ...
