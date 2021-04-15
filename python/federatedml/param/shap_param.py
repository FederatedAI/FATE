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


class ExplainableParam(BaseParam):

    def __init__(self, need_explain=False, method='shap', interpret_limit=10, shap_subset_sample_num='auto',
                 random_seed=100, ):
        super(ExplainableParam, self).__init__()
        self.need_explain = need_explain
        self.method = method
        self.interpret_limit = interpret_limit
        self.shap_subset_sample_num = shap_subset_sample_num
        self.random_seed = random_seed

    def check(self):
        self.check_boolean(self.need_explain, 'need_explain')
        self.check_positive_integer(self.interpret_limit, 'interpret_limit')
        self.check_positive_integer(self.random_seed, 'random_seed')


