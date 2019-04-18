import functools
from collections import Iterable

from federatedml.statistic.statics import MultivariateStatisticalSummary
from federatedml.util import fate_operator
from federatedml.statistic import data_overview


class StandardScaler(object):
    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std

    @staticmethod
    def __scale(data, mean, std):
        for i in range(data.features.shape[0]):
            data.features[i] = (data.features[i] - mean[i]) / std[i]

        return data

    @staticmethod
    def __scale_with_value(data, mean, std):
        for i in range(data.features.shape[0]):
            data.features[i] = (data.features[i] - mean) / std

        return data

    def fit(self, data):
        if not self.with_mean and not self.with_std:
            shape = data_overview.get_features_shape(data)
            mean = [0 for _ in range(shape)]
            std = [1 for _ in range(shape)]
            return data, mean, std

        else:
            summary_obj = MultivariateStatisticalSummary(data, -1)
            mean = None
            std = None

            if self.with_mean:
                mean = summary_obj.get_mean()

            if self.with_std:
                std = summary_obj.get_std_variance()

            if not mean and std:
                mean = [0 for value in std]
            elif mean and not std:
                std = [1 for value in mean]

            if not mean or not std:
                raise ValueError("mean or std is None")

            f = functools.partial(self.__scale, mean=mean, std=std)
            data = data.mapValues(f)

            return data, mean, std

    def transform(self, data, mean, scale):
        if isinstance(mean, Iterable) and isinstance(scale, Iterable):
            f = functools.partial(self.__scale, mean=mean, std=scale)
        elif not isinstance(mean, Iterable) and not isinstance(scale, Iterable):
            f = functools.partial(self.__scale_with_value, mean=mean, std=scale)
        else:
            raise ValueError("mean and scale should be all Iterable or all not Iterable")
        return data.mapValues(f)