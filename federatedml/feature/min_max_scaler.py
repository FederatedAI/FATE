import functools
import numpy as np
from collections import Iterable
from federatedml.statistic.statics import MultivariateStatisticalSummary
# from federatedml.util import fate_operator
from federatedml.statistic import data_overview


class MinMaxScaler(object):
    def __init__(self, mode='normal', area='all', feat_upper=None, feat_lower=None, out_upper=None, out_lower=None):
        self.mode = mode
        self.area = area
        self.feat_upper = feat_upper
        self.feat_lower = feat_lower
        self.out_upper = out_upper
        self.out_lower = out_lower

    def __get_min_max_value(self, data):
        min_value = None
        max_value = None
        summary_obj = MultivariateStatisticalSummary(data, -1)

        if self.feat_upper is not None:
            max_value = self.feat_upper

        if self.feat_lower is not None:
            min_value = self.feat_lower

        if min_value is None and max_value is not None:
            min_value_list = summary_obj.get_min()

            if isinstance(max_value, Iterable):
                if len(list(max_value)) != len(min_value_list):
                    raise ValueError(
                        "Size of feat_upper is not equal to column of data, {} != {}".format(len(list(max_value)),
                                                                                             len(min_value_list)))
                max_value_list = max_value
            else:
                max_value_list = [max_value for v in min_value_list]

        elif min_value is not None and max_value is None:
            max_value_list = summary_obj.get_max()

            if isinstance(min_value, Iterable):
                if len(list(min_value)) != len(max_value_list):
                    raise ValueError(
                        "Size of feat_lower is not equal to column of data, {} != {}".format(len(list(max_value)),
                                                                                             len(max_value_list)))
                min_value_list = min_value
            else:
                min_value_list = [min_value for v in max_value_list]

        elif min_value is None and max_value is None:
            min_value_list = summary_obj.get_min()
            max_value_list = summary_obj.get_max()
        else:
            shape = None
            if isinstance(max_value, Iterable):
                max_value_list = max_value
            else:
                shape = data_overview.get_features_shape(data)
                max_value_list = [max_value for _ in range(shape)]

            if isinstance(min_value, Iterable):
                min_value_list = min_value
            else:
                if not shape:
                    shape = data_overview.get_features_shape(data)

                min_value_list = [min_value for _ in range(shape)]

            if len(list(max_value_list)) != len(min_value_list):
                raise ValueError(
                    "Size of feat_upper is not equal to column of data, {} != {}".format(len(list(max_value)),
                                                                                         len(min_value_list)))

        return min_value_list, max_value_list

    def __get_upper_lower_percentile(self, data):
        pass

    def __check_param(self):
        support_mode = ['normal', 'cap']
        support_area = ['col', 'all']
        if self.mode not in support_mode:
            raise ValueError("Unsupport mode:{}".format(self.mode))

        if self.area not in support_area:
            raise ValueError("Unsupport area:{}".format(self.area))

        check_value_list = [self.feat_upper, self.feat_lower]
        for check_value in check_value_list:
            if check_value is not None:
                if self.area == 'all':
                    if not isinstance(check_value, int) and not isinstance(check_value, float):
                        raise ValueError(
                            "for area is all, {} should be int or float, not {}".format(check_value, type(check_value)))

                elif self.area == 'col':
                    if not isinstance(check_value, Iterable):
                        raise ValueError(
                            "for area is col, {} should be Iterable, not {}".format(check_value, type(check_value)))

        check_value_list = [self.out_upper, self.out_lower]
        for check_value in check_value_list:
            if check_value is not None:
                if not isinstance(check_value, int) and not isinstance(check_value, float):
                    raise ValueError(
                        "for area is all, {} should be int or float, not {}".format(check_value, type(check_value)))

        if self.feat_upper is not None and self.feat_lower is not None:
            if self.area == 'all':
                if float(self.feat_upper) < float(self.feat_lower):
                    raise ValueError("for area is all, feat_upper should not less than feat_lower, but {} < {}".format(
                        self.feat_upper, self.feat_lower))
            elif self.area == 'col':
                if len(list(self.feat_upper)) != len(list(self.feat_lower)):
                    raise ValueError(
                        "for area is col, sizeof feat_upper should equal to the sizeof feat_lower, but {} != {}".format(
                            len(list(self.feat_upper)), len(list(self.feat_lower))))

                for i in range(len(list(self.feat_upper))):
                    if float(self.feat_upper[i]) < float(self.feat_lower[i]):
                        raise ValueError(
                            "for area is col, feat_upper[{}] should not less than feat_lower[{}], but {} < {}".format(i,
                                                                                                                      i,
                                                                                                                      self.feat_upper[
                                                                                                                          i],
                                                                                                                      self.feat_lower[
                                                                                                                          i]))

        if self.out_upper is not None and self.out_lower is not None:
            if float(self.out_upper) < float(self.out_lower):
                raise ValueError(
                    "for area is all, out_upper should not less than out_lower, but {} < {}".format(self.out_upper,
                                                                                                    self.out_lower))

    @staticmethod
    def __scale_with_cols_for_instance(data, max_value_list, min_value_list, scale_value_list, out_lower, out_scale):
        for i in range(data.features.shape[0]):
            if data.features[i] > max_value_list[i]:
                value = 1
            elif data.features[i] < min_value_list[i]:
                value = 0
            else:
                value = (data.features[i] - min_value_list[i]) / scale_value_list[i]

            data.features[i] = np.around(value * out_scale + out_lower, 4)

        return data

    def fit(self, data):
        self.__check_param()

        # if self.mode == 'normal':
        min_value, max_value = self.__get_min_max_value(data)

        out_lower = 0 if self.out_lower is None else self.out_lower
        out_upper = 1 if self.out_upper is None else self.out_upper

        out_scale = out_upper - out_lower
        if np.abs(out_scale - 0) < 1e-6 or out_scale < 0:
            raise ValueError("out_scale should large than 0")

        cols_transform_value = []
        data_scale = []

        for i in range(len(max_value)):
            scale = max_value[i] - min_value[i]
            if np.abs(scale - 0) < 1e-6 or scale < 0:
                raise ValueError("scale value should large than 0")
            data_scale.append(scale)
            cols_transform_value.append((min_value[i], max_value[i], out_lower, out_upper))

        f = functools.partial(MinMaxScaler.__scale_with_cols_for_instance, max_value_list=max_value,
                              min_value_list=min_value, scale_value_list=data_scale, out_lower=out_lower,
                              out_scale=out_scale)
        fit_data = data.mapValues(f)

        return fit_data, cols_transform_value

    def transform(self, data, cols_transform_value):
        max_value = []
        min_value = []
        out_upper = []
        out_lower = []
        data_scale = []

        for col in cols_transform_value:
            min_value.append(col[0])
            max_value.append(col[1])
            out_lower.append(col[2])
            out_upper.append(col[3])

            scale = col[1] - col[0]
            if np.abs(scale - 0) < 1e-6 or scale < 0:
                raise ValueError("scale value should large than 0")
            data_scale.append(scale)

        # check if each value of out_upper or out_lower is same
        if len(cols_transform_value) > 0:
            for i in range(len(cols_transform_value)):
                if out_lower[0] != out_lower[i] or out_upper[0] != out_upper[i]:
                    raise ValueError("In transform out_lower or out_upper have not same value")
        else:
            raise ValueError("cols_transform_value's size should larger than 0")

        out_scale = out_upper[0] - out_lower[0]
        if np.abs(out_scale - 0) < 1e-6 or out_scale < 0:
            raise ValueError("out_scale should large than 0")

        f = functools.partial(MinMaxScaler.__scale_with_cols_for_instance, max_value_list=max_value,
                              min_value_list=min_value, scale_value_list=data_scale, out_lower=out_lower[0],
                              out_scale=out_scale)

        fit_data = data.mapValues(f)
        return fit_data
