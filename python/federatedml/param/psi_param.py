from federatedml.param.base_param import BaseParam
from federatedml.util import consts


class PSIParam(BaseParam):

    def __init__(self, max_bin_num=20, need_run=True, dense_missing_val=None,
                 binning_error=consts.DEFAULT_RELATIVE_ERROR):
        super(PSIParam, self).__init__()
        self.max_bin_num = max_bin_num
        self.need_run = need_run
        self.dense_missing_val = dense_missing_val
        self.binning_error = binning_error

    def check(self):
        assert isinstance(self.max_bin_num, int) and self.max_bin_num > 0, 'max bin must be an integer larger than 0'
        assert isinstance(self.need_run, bool)

        if self.dense_missing_val is not None:
            assert isinstance(self.dense_missing_val, str) or isinstance(self.dense_missing_val, int) or \
                isinstance(self.dense_missing_val, float), \
                'missing value type {} not supported'.format(type(self.dense_missing_val))

        self.check_decimal_float(self.binning_error, "psi's param")
