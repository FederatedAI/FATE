import functools
from collections import Iterable

from arch.api.utils import log_utils
from federatedml.statistic.statics import MultivariateStatisticalSummary
from federatedml.statistic.data_overview import get_header
from federatedml.statistic import data_overview

LOGGER = log_utils.getLogger()


class StandardScaler(object):
    """
    Standardize features by removing the mean and scaling to unit variance. The standard score of a sample x is calculated as:
    z = (x - u) / s, where u is the mean of the training samples, and s is the standard deviation of the training samples
    """

    def __init__(self, area='all', scale_column_idx=None, with_mean=True, with_std=True):
        """
        Parameters
        ----------
        with_mean: bool, if true, the scaler will use the mean of the column and if false, mean will be zero
        with_std: bool, if true, the scaler will use the standard deviation of the column and if false, standard deviation will be one
        """
        self.area = area
        self.scale_column_idx = scale_column_idx
        self.with_mean = with_mean
        self.with_std = with_std

    @staticmethod
    def __scale(data, mean, std, process_cols_list):
        for i in process_cols_list:
            data.features[i] = (data.features[i] - mean[i]) / std[i]

        return data

    @staticmethod
    def __scale_with_value(data, mean, std):
        for i in range(data.features.shape[0]):
            data.features[i] = (data.features[i] - mean) / std

        return data

    def fit(self, data):
        """
         Apply standard scale for input data
         Parameters
         ----------
         data: data_instance, input data

         Returns
         ----------
         data:data_instance, data after scale
         mean: list, each column mean value
         std: list, each column standard deviation
         """
        if not self.with_mean and not self.with_std:
            shape = data_overview.get_features_shape(data)
            mean = [0 for _ in range(shape)]
            std = [1 for _ in range(shape)]
            self.scale_column_idx = [i for i in range(shape)]
            standard_scale_cols_conf = [ mean, std, self.scale_column_idx ]
            return data, standard_scale_cols_conf
        else:
            data_shape = data_overview.get_features_shape(data)
            if self.area == 'col':
                if isinstance(self.scale_column_idx, list):
                    max_col_idx = max(self.scale_column_idx)
                    if max_col_idx >= data_shape:
                        raise ValueError(
                            "max column index in area is:{}, should less than data shape:{}".format(max_col_idx,
                                                                                                    data_shape))
                    self.scale_column_idx.sort()
                else:
                    LOGGER.warning(
                        "scale_column_idx should be a list, but not:{}, set scale column to all columns".format(
                            type(self.scale_column_idx)))
                    self.scale_column_idx = [i for i in range(data_shape)]
            else:
                self.scale_column_idx = [i for i in range(data_shape)]

            summary_obj = MultivariateStatisticalSummary(data, -1)
            mean = None
            std = None
            header = get_header(data)

            if self.with_mean:
                mean = summary_obj.get_mean()
                mean = [mean[key] for key in header]

            if self.with_std:
                std = summary_obj.get_std_variance()
                std = [std[key] for key in header]

            if not mean and std:
                mean = [0 for _ in std]
            elif mean and not std:
                std = [1 for _ in mean]

            if not mean or not std:
                raise ValueError("mean or std is None")

            f = functools.partial(self.__scale, mean=mean, std=std, process_cols_list=self.scale_column_idx)
            data = data.mapValues(f)

            standard_scale_cols_conf = [mean, std, self.scale_column_idx]

            return data, standard_scale_cols_conf

    def transform(self, data, mean, scale, scale_column_idx):
        """
        Transform input data using standard scale with fit results
        Parameters
        ----------
        data: data_instance, input data
        mean: list, each column mean value
        scale: list, each column standard deviation
        Returns
        ----------
        transform_data:data_instance, data after transform
        """
        if isinstance(mean, Iterable) and isinstance(scale, Iterable):
            f = functools.partial(self.__scale, mean=mean, std=scale, process_cols_list=scale_column_idx)
        else:
            raise ValueError("mean and scale should be all Iterable or all not Iterable")
        return data.mapValues(f)
