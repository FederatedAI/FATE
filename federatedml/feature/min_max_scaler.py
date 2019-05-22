import functools
import numpy as np
from collections import Iterable
from federatedml.statistic.statics import MultivariateStatisticalSummary
from federatedml.statistic.data_overview import get_header
from federatedml.statistic import data_overview


class MinMaxScaler(object):
    """
    Transforms features by scaling each feature to a given range,e.g.between minimum and maximum. The transformation is given by:
            X_scale = (X - X.min) / (X.max - X.min), while X.min is the minimum value of feature, and X.max is the maximum
    """
    def __init__(self, mode='normal', area='all', feat_upper=None, feat_lower=None, out_upper=None, out_lower=None):
        """
        Parameters
        ----------
        mode: str, the mode just support "normal" now, and will support "cap" mode in the future.
              for mode is "normal", the feat_upper and feat_lower is the normal value and for "cap", feat_upper and
              feature_lower will between 0 and 1, which means the percentile of the column. Default "normal"

        area: str. It supports "all" and "col". For "all",feat_upper/feat_lower will act on all data column,
            so it will just be a value, and for "col", it just acts on one column they corresponding to,
            so feat_lower/feat_upper will be a list, which size will equal to the number of columns

        feat_upper: int or float, the upper limit in the column. If the value is larger than feat_upper, it will be set to feat_upper. Default None.
        feat_lower: int or float, the lower limit in the column. If the value is less than feat_lower, it will be set to feat_lower. Default None.
        out_upper: int or float,  the results of scale will be mapped to the area between out_lower and out_upper.Default None.
        out_upper: int or float, the results of scale will be mapped to the area between out_lower and out_upper.Default None.
        """
        self.mode = mode
        self.area = area
        self.feat_upper = feat_upper
        self.feat_lower = feat_lower
        self.out_upper = out_upper
        self.out_lower = out_lower

    def __get_min_max_value(self, data):
        """
        Get each column minimum and maximum
        """
        min_value = None
        max_value = None
        summary_obj = MultivariateStatisticalSummary(data, -1)
        header = get_header(data)

        if self.feat_upper is not None:
            max_value = self.feat_upper

        if self.feat_lower is not None:
            min_value = self.feat_lower

        if min_value is None and max_value is not None:
            min_value_dict = summary_obj.get_min()
            min_value_list = [ min_value_dict[key] for key in header ]

            if isinstance(max_value, Iterable):
                if len(list(max_value)) != len(min_value_list):
                    raise ValueError(
                        "Size of feat_upper is not equal to column of data, {} != {}".format(len(list(max_value)),
                                                                                             len(min_value_list)))
                max_value_list = max_value
            else:
                max_value_list = [max_value for _ in min_value_list]

        elif min_value is not None and max_value is None:
            max_value_dict = summary_obj.get_max()
            max_value_list = [max_value_dict[key] for key in header]

            if isinstance(min_value, Iterable):
                if len(list(min_value)) != len(max_value_list):
                    raise ValueError(
                        "Size of feat_lower is not equal to column of data, {} != {}".format(len(list(max_value)),
                                                                                             len(max_value_list)))
                min_value_list = min_value
            else:
                min_value_list = [min_value for _ in max_value_list]

        elif min_value is None and max_value is None:
            min_value_dict = summary_obj.get_min()
            max_value_dict = summary_obj.get_max()
            min_value_list = [min_value_dict[key] for key in header]
            max_value_list = [max_value_dict[key] for key in header]
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
        # TODO
        pass

    def __check_param(self):
        """
        Check if input parameter is legal
        """
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
        """
        Scale operator for each column. The input data type is data_instance
        """
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
        """
        Apply min-max scale for input data
        Parameters
        ----------
        data: data_instance, input data

        Returns
        ----------
        fit_data:data_instance, data after scale
        cols_transform_value: list of tuple, each tuple include minimum, maximum, output_minimum, output maximum
        """
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
        """
        Transform input data using min-max scale with fit results
        Parameters
        ----------
        data: data_instance, input data
        cols_transform_value: list of tuple, the return of fit function. Each tuple include minimum, maximum, output_minimum, output maximum
        Returns
        ----------
        transform_data:data_instance, data after transform
        """
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
