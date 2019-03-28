import functools
from collections import Iterable

import numpy as np


# from federatedml.feature import Instance

class MinMaxScale(object):
    def __init__(self, mode='normal', area='all', feat_upper=None, feat_lower=None, out_upper=None, out_lower=None):
        self.mode = mode
        self.area = area
        self.feat_upper = feat_upper
        self.feat_lower = feat_lower
        self.out_upper = out_upper
        self.out_lower = out_lower

    @staticmethod
    def __get_min(data):
        min_list = None
        for key, value in data:
            if min_list is None:
                min_list = [None for i in range(len(value))]

            for i in range(len(value)):
                try:
                    f_value = float(value[i])
                except:
                    f_value = None

                if f_value is None:
                    continue

                if min_list[i] is None or f_value < min_list[i]:
                    min_list[i] = f_value

        return min_list

    @staticmethod
    def __get_min_for_instance(data):
        min_list = None
        for key, value in data:
            if min_list is None:
                min_list = [None for i in range(len(value.features))]

            for i in range(len(value.features)):
                try:
                    f_value = float(value.features[i])
                except:
                    f_value = None

                if f_value is None:
                    continue

                if min_list[i] is None or f_value < min_list[i]:
                    min_list[i] = f_value

        return min_list

    @staticmethod
    def __get_max(data):
        max_list = None
        for key, value in data:
            if max_list is None:
                max_list = [None for i in range(len(value))]

            for i in range(len(value)):
                try:
                    f_value = float(value[i])
                except:
                    f_value = None

                if f_value is None:
                    continue

                if max_list[i] is None or f_value > max_list[i]:
                    max_list[i] = f_value

        return max_list

    @staticmethod
    def __get_max_for_instance(data):
        max_list = None
        for key, value in data:
            if max_list is None:
                max_list = [None for i in range(len(value.features))]

            for i in range(len(value.features)):
                try:
                    f_value = float(value.features[i])
                except:
                    f_value = None

                if f_value is None:
                    continue

                if max_list[i] is None or f_value > max_list[i]:
                    max_list[i] = f_value

        return max_list

    def __get_cols_transform_min_value(self, data):
        min_lists = data.mapPartitions(MinMaxScale.__get_min_for_instance)

        cols_transform_min_value = None
        for min_tuple in list(min_lists.collect()):
            if cols_transform_min_value is None:
                cols_transform_min_value = min_tuple[1]
            else:
                # some return of partition maybe None
                if min_tuple[1] is None:
                    continue

                for i in range(len(min_tuple[1])):
                    if min_tuple[1][i] is None:
                        continue

                    if cols_transform_min_value[i] is None:
                        cols_transform_min_value[i] = min_tuple[1][i]
                    elif min_tuple[1][i] < cols_transform_min_value[i]:
                        cols_transform_min_value[i] = min_tuple[1][i]
        return cols_transform_min_value

    def __get_cols_transform_max_value(self, data):
        max_lists = data.mapPartitions(MinMaxScale.__get_max_for_instance)

        cols_transform_max_value = None
        for max_tuple in list(max_lists.collect()):
            if cols_transform_max_value is None:
                cols_transform_max_value = max_tuple[1]
            else:
                # some return of partition maybe None
                if max_tuple[1] is None:
                    continue

                for i in range(len(max_tuple[1])):
                    if max_tuple[1][i] is None:
                        continue

                    if cols_transform_max_value[i] is None:
                        cols_transform_max_value[i] = max_tuple[1][i]
                    elif max_tuple[1][i] > cols_transform_max_value[i]:
                        cols_transform_max_value[i] = max_tuple[1][i]
        return cols_transform_max_value

    def __get_min_max_value(self, data):
        min_value = None
        max_value = None
        if self.feat_upper != None:
            max_value = self.feat_upper

        if self.feat_lower != None:
            min_value = self.feat_lower

        if min_value is None and max_value is not None:
            min_value_list = self.__get_cols_transform_min_value(data)
            if isinstance(max_value, Iterable):
                if len(list(max_value)) != len(min_value_list):
                    raise ValueError(
                        "Size of feat_upper is not equal to column of data, {} != {}".format(len(list(max_value)),
                                                                                             len(min_value_list)))
                max_value_list = max_value
            else:
                max_value_list = [max_value for v in min_value_list]

        elif min_value is not None and max_value is None:
            max_value_list = self.__get_cols_transform_max_value(data)
            if isinstance(min_value, Iterable):
                if len(list(min_value)) != len(max_value_list):
                    raise ValueError(
                        "Size of feat_lower is not equal to column of data, {} != {}".format(len(list(max_value)),
                                                                                             len(max_value_list)))
                min_value_list = min_value
            else:
                min_value_list = [min_value for v in max_value_list]

        elif min_value is None and max_value is None:
            min_value_list = self.__get_cols_transform_min_value(data)
            max_value_list = self.__get_cols_transform_max_value(data)
        else:
            min_value_list = min_value
            max_value_list = max_value

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
    def __scale_with_cols(data, max_value_list, min_value_list, scale_value_list, out_lower, out_scale):
        for i in range(len(data)):
            if data[i] > max_value_list[i]:
                value = 1
            elif data[i] < min_value_list[i]:
                value = 0
            else:
                value = (data[i] - min_value_list[i]) / scale_value_list[i]

            data[i] = np.around(value * out_scale + out_lower, 4)

        return data

    @staticmethod
    def __scale_with_value(data, max_value, min_value, scale_value, out_lower, out_scale):
        for i in range(len(data)):
            if data[i] > max_value:
                value = 1
            elif data[i] < min_value:
                value = 0
            else:
                value = (data[i] - min_value) / scale_value

            data[i] = np.around(value * out_scale + out_lower, 4)

        return data

    @staticmethod
    def __scale_with_cols_for_instance(data, max_value_list, min_value_list, scale_value_list, out_lower, out_scale):
        for i in range(len(data.features)):
            if data.features[i] > max_value_list[i]:
                value = 1
            elif data.features[i] < min_value_list[i]:
                value = 0
            else:
                value = (data.features[i] - min_value_list[i]) / scale_value_list[i]

            data.features[i] = np.around(value * out_scale + out_lower, 4)

        return data

    @staticmethod
    def __scale_with_value_for_instance(data, max_value, min_value, scale_value, out_lower, out_scale):
        for i in range(len(data.features)):
            if data.features[i] > max_value:
                value = 1
            elif data.features[i] < min_value:
                value = 0
            else:
                value = (data.features[i] - min_value) / scale_value

            data.features[i] = np.around(value * out_scale + out_lower, 4)

        return data

    def fit(self, data):
        self.__check_param()

        if self.mode == 'normal':
            min_value, max_value = self.__get_min_max_value(data)
        elif self.mode == 'cap':
            min_value, max_value = self.__get_upper_lower_percentile(data)

        out_lower = 0 if self.out_lower is None else self.out_lower
        out_upper = 1 if self.out_upper is None else self.out_upper

        out_scale = out_upper - out_lower
        if np.abs(out_scale - 0) < 1e-6 or out_scale < 0:
            raise ValueError("out_scale should large than 0")

        cols_transform_value = []
        if not isinstance(max_value, Iterable) and not isinstance(min_value, Iterable):
            data_scale = max_value - min_value
            if np.abs(data_scale - 0) < 1e-6 or data_scale < 0:
                raise ValueError("scale value should large than 0")

            f = functools.partial(MinMaxScale.__scale_with_value_for_instance, max_value=max_value, min_value=min_value,
                                  scale_value=data_scale, out_lower=out_lower, out_scale=out_scale)
            cols_transform_value.append((min_value, max_value, out_lower, out_upper))
        elif isinstance(max_value, Iterable) and isinstance(min_value, Iterable):
            data_scale = []
            for i in range(len(max_value)):
                scale = max_value[i] - min_value[i]
                if np.abs(scale - 0) < 1e-6 or scale < 0:
                    raise ValueError("scale value should large than 0")
                data_scale.append(scale)
                cols_transform_value.append((min_value[i], max_value[i], out_lower, out_upper))

            f = functools.partial(MinMaxScale.__scale_with_cols_for_instance, max_value_list=max_value,
                                  min_value_list=min_value, scale_value_list=data_scale, out_lower=out_lower,
                                  out_scale=out_scale)
        else:
            raise ValueError("max has not same size with min")

        fit_data = data.mapValues(f)
        return fit_data, cols_transform_value

    def transform(self, data, cols_transform_value):
        if len(cols_transform_value) == 1:
            if len(cols_transform_value[0]) != 4:
                raise ValueError("if cols_transform_value size is 1, it should has 4 elements, but {}".format(
                    len(cols_transform_value[0])))

            min_value = cols_transform_value[0][0]
            max_value = cols_transform_value[0][1]
            out_lower = cols_transform_value[0][2]
            out_upper = cols_transform_value[0][3]

            data_scale = max_value - min_value
            if np.abs(data_scale - 0) < 1e-6 or data_scale < 0:
                raise ValueError("scale value should large than 0")

            out_scale = out_upper - out_lower
            if np.abs(out_scale - 0) < 1e-6 or out_scale < 0:
                raise ValueError("out_scale should large than 0")

            f = functools.partial(MinMaxScale.__scale_with_value_for_instance, max_value=max_value, min_value=min_value,
                                  scale_value=data_scale, out_lower=out_lower, out_scale=out_scale)
        else:
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

            f = functools.partial(MinMaxScale.__scale_with_cols_for_instance, max_value_list=max_value,
                                  min_value_list=min_value, scale_value_list=data_scale, out_lower=out_lower[0],
                                  out_scale=out_scale)

        fit_data = data.mapValues(f)
        return fit_data
