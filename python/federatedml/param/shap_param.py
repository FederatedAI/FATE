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


class SHAPParam(BaseParam):

    def __init__(self, interpret_limit=10, subset_sample_num='auto', random_seed=100, need_shap=False):
        super(SHAPParam, self).__init__()
        self.interpret_limit = interpret_limit
        self.subset_sample_num = subset_sample_num
        self.random_seed = random_seed
        self.need_shap = need_shap

    def check(self):
        self.check_positive_integer(self.interpret_limit, 'interpret_limit')
        self.check_positive_integer(self.random_seed, 'random_seed')

