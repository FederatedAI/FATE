from federatedml.param.base_param import BaseParam
from federatedml.util import consts


class TreeSHAPParam(BaseParam):

    def __init__(self, interpret_limit=10):
        super(TreeSHAPParam, self).__init__()
        self.interpret_limit = interpret_limit

    def check(self):
        self.check_positive_integer(self.interpret_limit, 'interpret limits')


class KernelSHAPParam(BaseParam):

    def __init__(self, interpret_limit=10):
        super(KernelSHAPParam, self).__init__()
        self.interpret_limit = interpret_limit

    def check(self):
        self.check_positive_integer(self.interpret_limit, 'interpret limits')

