from federatedml.param.base_param import BaseParam
from federatedml.util import consts


class ModelInterpretParam(BaseParam):

    def __init__(self, need_explain=False, method='shap', interpret_limit=10, shap_subset_sample_num='auto'
                 ,reference_type='all_zero', random_seed=100, ):
        super(ModelInterpretParam, self).__init__()
        self.need_explain = need_explain
        self.method = method
        self.interpret_limit = interpret_limit
        self.shap_subset_sample_num = shap_subset_sample_num
        self.random_seed = random_seed
        self.reference_type = reference_type

    def check(self):
        self.check_boolean(self.need_explain, 'need_explain')
        self.check_positive_integer(self.interpret_limit, 'interpret_limit')
        self.check_positive_integer(self.random_seed, 'random_seed')
        if self.shap_subset_sample_num != 'auto':
            self.check_positive_integer(self.shap_subset_sample_num, 'shap_subset_sample_num')


