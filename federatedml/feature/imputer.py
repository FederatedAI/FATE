import copy
import functools
import numpy as np

from federatedml.statistic.data_overview import get_header
from federatedml.statistic.statics import MultivariateStatisticalSummary
from federatedml.util import consts
from federatedml.util import LOGGER
from federatedml.statistic import data_overview


class Imputer(object):
    """
    This class provides basic strategies for values replacement. It can be used as missing filled or outlier replace.
    You can use the statistics such as mean, median or max of each column to fill the missing value or replace outlier.
    """

    def __init__(self, missing_value_list=None):
        """
        Parameters
        ----------
        missing_value_list: list of str, the value to be replaced. Default None, if is None, it will be set to list of blank, none, null and na,
                            which regarded as missing filled. If not, it can be outlier replace, and missing_value_list includes the outlier values
        """
        if missing_value_list is None:
            self.missing_value_list = ['', 'none', 'null', 'na']
        else:
            self.missing_value_list = missing_value_list

        self.support_replace_method = ['min', 'max', 'mean', 'median', 'quantile', 'designated']
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
            'median': 'col',
            'quantile': 'col',
            'designated': 'col'
        }

        self.cols_fit_impute_rate = []
        self.cols_transform_impute_rate = []

    def get_missing_value_list(self):
        return self.missing_value_list

    def get_impute_rate(self, mode="fit"):
        if mode == "fit":
            return list(self.cols_fit_impute_rate)
        elif mode == "transform":
            return list(self.cols_transform_impute_rate)
        else:
            raise ValueError("Unknown mode of {}".format(mode))

    @staticmethod
    def __replace_missing_value_with_cols_transform_value_format(data, transform_list, missing_value_list,
                                                                 output_format):
        _data = copy.deepcopy(data)
        replace_cols_index_list = []
        for i, v in enumerate(_data):
            if str(v) in missing_value_list:
                _data[i] = output_format(transform_list[i])
                replace_cols_index_list.append(i)
            else:
                _data[i] = output_format(v)

        return _data, replace_cols_index_list

    @staticmethod
    def __replace_missing_value_with_cols_transform_value(data, transform_list, missing_value_list):
        _data = copy.deepcopy(data)
        replace_cols_index_list = []
        for i, v in enumerate(_data):
            if str(v) in missing_value_list:
                _data[i] = str(transform_list[i])
                replace_cols_index_list.append(i)

        return _data, replace_cols_index_list

    @staticmethod
    def __replace_missing_value_with_replace_value_format(data, replace_value, missing_value_list, output_format):
        _data = copy.deepcopy(data)
        replace_cols_index_list = []
        for i, v in enumerate(_data):
            if str(v) in missing_value_list:
                _data[i] = output_format(replace_value)
                replace_cols_index_list.append(i)
            else:
                _data[i] = output_format(_data[i])

        return _data, replace_cols_index_list

    @staticmethod
    def __replace_missing_value_with_replace_value(data, replace_value, missing_value_list):
        _data = copy.deepcopy(data)
        replace_cols_index_list = []
        for i, v in enumerate(_data):
            if str(v) in missing_value_list:
                _data[i] = str(replace_value)
                replace_cols_index_list.append(i)

        return _data, replace_cols_index_list

    def __get_cols_transform_value(self, data, replace_method, quantile=None):
        summary_obj = MultivariateStatisticalSummary(data, -1, abnormal_list=self.missing_value_list)
        header = get_header(data)

        if replace_method == consts.MIN:
            cols_transform_value = summary_obj.get_min()
        elif replace_method == consts.MAX:
            cols_transform_value = summary_obj.get_max()
        elif replace_method == consts.MEAN:
            cols_transform_value = summary_obj.get_mean()
        elif replace_method == consts.MEDIAN:
            cols_transform_value = summary_obj.get_median()
        elif replace_method == consts.QUANTILE:
            if quantile > 1 or quantile < 0:
                raise ValueError("quantile should between 0 and 1, but get:{}".format(quantile))
            cols_transform_value = summary_obj.get_quantile_point(quantile)
        else:
            raise ValueError("Unknown replace method:{}".format(replace_method))

        cols_transform_value = [round(cols_transform_value[key], 6) for key in header]
        return cols_transform_value

    def __fit_replace(self, data, replace_method, replace_value=None, output_format=None, quantile=None):
        if replace_method is not None and replace_method != consts.DESIGNATED:
            cols_transform_value = self.__get_cols_transform_value(data, replace_method, quantile=quantile)
            if output_format is not None:
                f = functools.partial(Imputer.__replace_missing_value_with_cols_transform_value_format,
                                      transform_list=cols_transform_value, missing_value_list=self.missing_value_list,
                                      output_format=output_format)
            else:
                f = functools.partial(Imputer.__replace_missing_value_with_cols_transform_value,
                                      transform_list=cols_transform_value, missing_value_list=self.missing_value_list)

            transform_data = data.mapValues(f)
            LOGGER.info(
                "finish replace missing value with cols transform value, replace method is {}".format(replace_method))
            return transform_data, cols_transform_value
        else:
            if replace_value is None:
                raise ValueError("Replace value should not be None")
            if output_format is not None:
                f = functools.partial(Imputer.__replace_missing_value_with_replace_value_format,
                                      replace_value=replace_value, missing_value_list=self.missing_value_list,
                                      output_format=output_format)
            else:
                f = functools.partial(Imputer.__replace_missing_value_with_replace_value, replace_value=replace_value,
                                      missing_value_list=self.missing_value_list)
            transform_data = data.mapValues(f)
            LOGGER.info("finish replace missing value with replace value {}, replace method is:{}".format(replace_value,
                                                                                                          replace_method))
            shape = data_overview.get_data_shape(data)
            replace_value = [replace_value for _ in range(shape)]

            return transform_data, replace_value

    def __transform_replace(self, data, transform_value, replace_area, output_format):
        if replace_area == 'all':
            if output_format is not None:
                f = functools.partial(Imputer.__replace_missing_value_with_replace_value_format,
                                      replace_value=transform_value, missing_value_list=self.missing_value_list,
                                      output_format=output_format)
            else:
                f = functools.partial(Imputer.__replace_missing_value_with_replace_value,
                                      replace_value=transform_value, missing_value_list=self.missing_value_list)
        elif replace_area == 'col':
            if output_format is not None:
                f = functools.partial(Imputer.__replace_missing_value_with_cols_transform_value_format,
                                      transform_list=transform_value, missing_value_list=self.missing_value_list,
                                      output_format=output_format)
            else:
                f = functools.partial(Imputer.__replace_missing_value_with_cols_transform_value,
                                      transform_list=transform_value, missing_value_list=self.missing_value_list)
        else:
            raise ValueError("Unknown replace area {} in Imputer".format(replace_area))

        return data.mapValues(f)

    @staticmethod
    def __get_impute_number(some_data):
        impute_num_list = None
        data_size = None

        for line in some_data:
            processed_data = line[1][0]
            index_list = line[1][1]
            if not data_size:
                data_size = len(processed_data)
                # data_size + 1, the last element of impute_num_list used to count the number of "some_data"
                impute_num_list = [0 for _ in range(data_size + 1)]

            impute_num_list[data_size] += 1
            for index in index_list:
                impute_num_list[index] += 1

        return np.array(impute_num_list)

    def __get_impute_rate_from_replace_data(self, data):
        impute_number_statics = data.applyPartitions(self.__get_impute_number).reduce(lambda x, y: x + y)
        cols_impute_rate = impute_number_statics[:-1] / impute_number_statics[-1]

        return cols_impute_rate

    def fit(self, data, replace_method=None, replace_value=None, output_format=consts.ORIGIN, quantile=None):
        """
        Apply imputer for input data
        Parameters
        ----------
        data: DTable, each data's value should be list
        replace_method: str, the strategy of imputer, like min, max, mean or designated and so on. Default None
        replace_value: str, if replace_method is designated, you should assign the replace_value which will be used to replace the value in imputer_value_list
        output_format: str, the output data format. The output data can be 'str', 'int', 'float'. Default origin, the original format as input data

        Returns
        ----------
        fit_data:data_instance, data after imputer
        cols_transform_value: list, the replace value in each column
        """
        if output_format not in self.support_output_format:
            raise ValueError("Unsupport output_format:{}".format(output_format))

        output_format = self.support_output_format[output_format]

        if isinstance(replace_method, str):
            replace_method = replace_method.lower()
            if replace_method not in self.support_replace_method:
                raise ValueError("Unknown replace method:{}".format(replace_method))
        elif replace_method is None:
            replace_value = '0'
        else:
            raise ValueError("parameter replace_method should be str or None only")

        process_data, cols_transform_value = self.__fit_replace(data, replace_method, replace_value, output_format,
                                                                quantile=quantile)

        self.cols_fit_impute_rate = self.__get_impute_rate_from_replace_data(process_data)
        process_data = process_data.mapValues(lambda v:v[0])
        process_data.schema = data.schema

        return process_data, cols_transform_value

    def transform(self, data, transform_value, output_format=consts.ORIGIN):
        """
        Transform input data using Imputer with fit results
        Parameters
        ----------
        data: DTable, each data's value should be list
        transform_value:
        output_format: str, the output data format. The output data can be 'str', 'int', 'float'. Default origin, the original format as input data

        Returns
        ----------
        transform_data:data_instance, data after transform
        """
        if output_format not in self.support_output_format:
            raise ValueError("Unsupport output_format:{}".format(output_format))

        output_format = self.support_output_format[output_format]

        # Now all of replace_method is "col", remain replace_area temporarily
        # replace_area = self.support_replace_area[replace_method]
        replace_area = "col"
        process_data = self.__transform_replace(data, transform_value, replace_area, output_format)
        self.cols_transform_impute_rate = self.__get_impute_rate_from_replace_data(process_data)
        process_data = process_data.mapValues(lambda v: v[0])
        process_data.schema = data.schema

        return process_data
