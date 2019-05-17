import functools
import numpy as np

from federatedml.util import consts
from federatedml.statistic import data_overview

from arch.api.utils import log_utils

LOGGER = log_utils.getLogger()


class Imputer(object):
    def __init__(self, imputer_value_list=None):
        if imputer_value_list is None:
            self.imputer_value_list = ['', 'none', 'null', 'na']
        else:
            self.imputer_value_list = imputer_value_list

        self.support_replace_method = ['min', 'max', 'mean', 'designated']
        # self.support_replace_method = ['min', 'max', 'mean', 'meadian', 'quantile', 'designated' ]
        self.support_output_format = {
            'str': str,
            'float': float,
            'int': int,
            'origin': None
        }

        self.support_replace_area = {
            'min': 'col',
            'max': 'col',
            'mean': 'col',
            'meadian': 'col',
            'quantile': 'col',
            'designated': 'col'
        }

    def get_imputer_value_list(self):
        return self.imputer_value_list

    @staticmethod
    def __get_min(data):
        min_list = None
        for key, value in data:
            if min_list is None:
                min_list = [None for _ in range(len(value))]

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
    def __get_mean(data):
        cols_value_sum = None
        cols_value_counter = None
        for key, value in data:
            if cols_value_sum is None:
                cols_value_sum = [0 for i in range(len(value))]
                cols_value_counter = [0 for i in range(len(value))]

            for i in range(len(value)):
                try:
                    f_value = float(value[i])
                except:
                    f_value = None

                if f_value is None:
                    continue

                cols_value_sum[i] += f_value
                cols_value_counter[i] += 1

        return cols_value_sum, cols_value_counter

    @staticmethod
    def __replace_missing_value_with_cols_transform_value_format(data, transform_list, missing_value_list,
                                                                 output_format):
        for i in range(len(data)):
            if str(data[i]).lower() in missing_value_list:
                data[i] = output_format(transform_list[i])
            else:
                data[i] = output_format(data[i])

        return data

    @staticmethod
    def __replace_missing_value_with_cols_transform_value(data, transform_list, missing_value_list):
        for i in range(len(data)):
            if str(data[i]).lower() in missing_value_list:
                data[i] = str(transform_list[i])

        return data

    @staticmethod
    def __replace_missing_value_with_replace_value_format(data, replace_value, missing_value_list, output_format):
        for i in range(len(data)):
            if str(data[i]).lower() in missing_value_list:
                data[i] = output_format(replace_value)
            else:
                data[i] = output_format(data[i])

        return data

    @staticmethod
    def __replace_missing_value_with_replace_value(data, replace_value, missing_value_list):
        for i in range(len(data)):
            if str(data[i]).lower() in missing_value_list:
                data[i] = str(replace_value)

        return data

    def __get_cols_transform_min_value(self, data):
        min_lists = data.mapPartitions(Imputer.__get_min)
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
        max_lists = data.mapPartitions(Imputer.__get_max)
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

    def __get_cols_transform_mean_value(self, data):
        get_mean_results = data.mapPartitions(Imputer.__get_mean)
        cols_sum = None
        cols_counter = None

        for value_tuple in list(get_mean_results.collect()):
            if cols_sum is None:
                cols_sum = [0 for i in range(len(value_tuple[1][0]))]
            if cols_counter is None:
                cols_counter = [0 for i in range(len(value_tuple[1][1]))]

            value_sum = value_tuple[1][0]
            value_counter = value_tuple[1][1]
            # some return of partition maybe None
            if value_sum is None and value_counter is None:
                continue

            for i in range(len(value_sum)):
                if value_sum[i] is None:
                    LOGGER.debug("col {} of cols_sum is None, continue".format(i))
                    continue

                cols_sum[i] += value_sum[i]

            for i in range(len(value_counter)):
                if value_counter[i] is None:
                    LOGGER.debug("col {} of cols_counter is None, continue".format(i))
                    continue

                cols_counter[i] += value_counter[i]

        if cols_sum is None or cols_counter is None:
            raise ValueError("Something wrong with data")

        cols_transform_mean_value = None
        for i in range(len(cols_sum)):
            if cols_sum[i] is None or cols_counter[i] is None:
                raise ValueError("Something wrong with cols_sum or cols_counter")

            if cols_transform_mean_value is None:
                cols_transform_mean_value = [None for i in range(len(cols_sum))]

            if cols_counter[i] == 0:
                cols_transform_mean_value[i] = 0
            else:
                cols_transform_mean_value[i] = np.around(cols_sum[i] / cols_counter[i], 6)

        if None in cols_transform_mean_value:
            raise ValueError("Some of value in cols_transform_mean_value is None, please check it")

        return cols_transform_mean_value

    def __get_cols_transform_value(self, data, replace_method='0'):
        if replace_method == consts.MIN:
            cols_transform_value = self.__get_cols_transform_min_value(data)
        elif replace_method == consts.MAX:
            cols_transform_value = self.__get_cols_transform_max_value(data)
        elif replace_method == consts.MEAN:
            cols_transform_value = self.__get_cols_transform_mean_value(data)
        else:
            raise ValueError("Unknown replace method:{}".format(replace_method))

        return cols_transform_value

    def __replace(self, data, replace_method, replace_value=None, output_format=None):
        if replace_method is not None and replace_method != consts.DESIGNATED:
            cols_transform_value = self.__get_cols_transform_value(data, replace_method)
            if output_format is not None:
                f = functools.partial(Imputer.__replace_missing_value_with_cols_transform_value_format,
                                      transform_list=cols_transform_value, missing_value_list=self.imputer_value_list,
                                      output_format=output_format)
            else:
                f = functools.partial(Imputer.__replace_missing_value_with_cols_transform_value,
                                      transform_list=cols_transform_value, missing_value_list=self.imputer_value_list)

            transform_data = data.mapValues(f)
            LOGGER.debug(
                "finish replace missing value with cols transform value, replace method is {}".format(replace_method))
            return transform_data, cols_transform_value
        else:
            if replace_value is None:
                raise ValueError("Replace value should not be None")
            if output_format is not None:
                f = functools.partial(Imputer.__replace_missing_value_with_replace_value_format,
                                      replace_value=replace_value, missing_value_list=self.imputer_value_list,
                                      output_format=output_format)
            else:
                f = functools.partial(Imputer.__replace_missing_value_with_replace_value, replace_value=replace_value,
                                      missing_value_list=self.imputer_value_list)
            transform_data = data.mapValues(f)

            LOGGER.debug("finish replace missing value with replace value {}".format(replace_value))
            shape = data_overview.get_data_shape(data)
            replace_value = [replace_value for _ in range(shape)]

            return transform_data, replace_value

    def __transform_replace(self, data, transform_value, replace_area, output_format):
        LOGGER.debug("replace_area:{}".format(replace_area))
        if replace_area == 'all':
            if output_format is not None:
                f = functools.partial(Imputer.__replace_missing_value_with_replace_value_format,
                                      replace_value=transform_value, missing_value_list=self.imputer_value_list,
                                      output_format=output_format)
            else:
                f = functools.partial(Imputer.__replace_missing_value_with_replace_value, 
                                      replace_value=transform_value, missing_value_list=self.imputer_value_list)
        elif replace_area == 'col':
            if output_format is not None:
                f = functools.partial(Imputer.__replace_missing_value_with_cols_transform_value_format,
                                      transform_list=transform_value, missing_value_list=self.imputer_value_list,
                                      output_format=output_format)
            else:
                f = functools.partial(Imputer.__replace_missing_value_with_cols_transform_value,
                                      transform_list=transform_value, missing_value_list=self.imputer_value_list)
        else:
            raise ValueError("Unknown replace area {} in Imputer".format(replace_area))

        transform_data = data.mapValues(f)
        return transform_data

    def fit(self, data, replace_method=None, replace_value=None, output_format=consts.ORIGIN):
        if output_format not in self.support_output_format:
            raise ValueError("Unsupport output_format:{}".format(output_format))

        output_format = self.support_output_format[output_format]

        if isinstance(replace_method, str):
            replace_method = replace_method.lower()
            if replace_method not in self.support_replace_method:
                raise ValueError("Unknown replace method in Imputer")

            process_data, cols_transform_value = self.__replace(data, replace_method, replace_value, output_format)
            return process_data, cols_transform_value
        elif replace_method is None:
            replace_value = '0'
            process_data, replace_value = self.__replace(data, replace_method, replace_value, output_format)
            return process_data, replace_value
        else:
            raise ValueError("parameter replace_method should be str or None only")

    def transform(self, data, replace_method=None, transform_value=None, output_format=consts.ORIGIN):
        if output_format not in self.support_output_format:
            raise ValueError("Unsupport output_format:{}".format(output_format))

        output_format = self.support_output_format[output_format]

        # Now all of replace_method is "col", remain replace_area temporarily
        LOGGER.debug("replace_method:{}".format(replace_method))
        # replace_area = self.support_replace_area[replace_method]
        replace_area = "col"
        process_data = self.__transform_replace(data, transform_value, replace_area, output_format)

        # if isinstance(replace_method, str):
        #     replace_method = replace_method.lower()
        #     if replace_method not in self.support_replace_method:
        #         raise ValueError("Unknown replace method {} in Imputer".format(replace_method))
        #
        #     if replace_method not in self.support_replace_area:
        #         raise ValueError("Unknown replace area of method {} in Imputer".format(replace_method))
        #
        #     replace_area = self.support_replace_area[replace_method]
        #     process_data = self.__transform_replace(data, transform_value, replace_area, output_format)
        # elif replace_method is None:
        #     replace_area = 'all'
        #     process_data = self.__transform_replace(data, transform_value, replace_area, output_format)
        #     return process_data
        # else:
        #     raise ValueError("parameter replace_method should be str or None only")

        return process_data
