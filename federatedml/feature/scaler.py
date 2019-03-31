from arch.api.utils import log_utils
from federatedml.feature.min_max_scaler import MinMaxScaler
from federatedml.feature.standard_scaler import StandardScaler
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class Scaler(object):
    def __init__(self, scale_param):
        self.scale_param = scale_param

    def fit(self, data):
        LOGGER.info("Start scale data fit ...")
        scale_value_results = []

        if self.scale_param.method == consts.MINMAXSCALE:
            min_max_scaler = MinMaxScaler(mode=self.scale_param.mode, area=self.scale_param.area,
                                          feat_upper=self.scale_param.feat_upper,
                                          feat_lower=self.scale_param.feat_lower,
                                          out_upper=self.scale_param.out_upper, out_lower=self.scale_param.out_lower)

            data, cols_scale_value = min_max_scaler.fit(data)
            scale_value_results.append(cols_scale_value)

        elif self.scale_param.method == consts.STANDARDSCALE:
            standard_scaler = StandardScaler(with_mean=self.scale_param.with_mean, with_std=self.scale_param.with_std)
            data, mean, std = standard_scaler.fit(data)
            scale_value_results.append(mean)
            scale_value_results.append(std)

        else:
            LOGGER.info("Scale method is {}, do nothing and return!".format(self.scale_param.method))

        LOGGER.info("End fit data ...")
        return data, scale_value_results

    def transform(self, data, fit_config):
        LOGGER.info("Start scale data transform ...")

        if len(fit_config) == 0:
            LOGGER.warning("length fit_config is 0, can not do transform, do nothing and return")

        if self.scale_param.method == consts.MINMAXSCALE:
            min_max_scaler = MinMaxScaler()
            data = min_max_scaler.transform(data, fit_config[0])
        elif self.scale_param.method == consts.STANDARDSCALE:
            standard_scaler = StandardScaler()
            data = standard_scaler.transform(data, mean=fit_config[0], scale=fit_config[1])
        else:
            LOGGER.info("DataTransform method is {}, do nothing and return!".format(self.scale_param.method))

        LOGGER.info("End transform data ...")

        return data