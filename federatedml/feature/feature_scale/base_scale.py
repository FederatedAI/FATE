#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import copy
import functools
from collections import Iterable

from federatedml.statistic import data_overview
from federatedml.statistic.data_overview import get_header
from federatedml.statistic.statics import MultivariateStatisticalSummary
from federatedml.util import consts
from federatedml.util import LOGGER


class BaseScale(object):
    def __init__(self, params):
        # self.area = params.area
        self.mode = params.mode
        self.param_scale_col_indexes = params.scale_col_indexes
        self.param_scale_names = params.scale_names
        self.feat_upper = params.feat_upper
        self.feat_lower = params.feat_lower
        self.data_shape = None
        self.header = None
        self.scale_column_idx = []

        self.summary_obj = None

        self.model_param_name = 'ScaleParam'
        self.model_meta_name = 'ScaleMeta'

        self.column_min_value = None
        self.column_max_value = None

        self.round_num = 6

    def _get_data_shape(self, data):
        if not self.data_shape:
            self.data_shape = data_overview.get_features_shape(data)

        return self.data_shape

    def _get_header(self, data):
        header = get_header(data)
        return header

    def _get_upper(self, data_shape):
        if isinstance(self.feat_upper, Iterable):
            return list(map(str, self.feat_upper))
        else:
            if self.feat_upper is None:
                return ["None" for _ in range(data_shape)]
            else:
                return [str(self.feat_upper) for _ in range(data_shape)]

    def _get_lower(self, data_shape):
        if isinstance(self.feat_lower, Iterable):
            return list(map(str, self.feat_lower))
        else:
            if self.feat_lower is None:
                return ["None" for _ in range(data_shape)]
            else:
                return [str(self.feat_lower) for _ in range(data_shape)]

    def _get_scale_column_idx(self, data):
        data_shape = self._get_data_shape(data)
        if self.param_scale_col_indexes != -1:
            if isinstance(self.param_scale_col_indexes, list):
                if len(self.param_scale_col_indexes) > 0:
                    max_col_idx = max(self.param_scale_col_indexes)
                    if max_col_idx >= data_shape:
                        raise ValueError(
                            "max column index in area is:{}, should less than data shape:{}".format(max_col_idx,
                                                                                                    data_shape))
                scale_column_idx = self.param_scale_col_indexes

                header = data_overview.get_header(data)

                scale_names = set(header).intersection(set(self.param_scale_names))
                idx_from_name = list(map(lambda n: header.index(n), scale_names))

                scale_column_idx = scale_column_idx + idx_from_name
                scale_column_idx = list(set(scale_column_idx))
                scale_column_idx.sort()
            else:
                LOGGER.warning(
                    "parameter scale_column_idx should be a list, but not:{}, set scale column to all columns".format(
                        type(self.param_scale_col_indexes)))
                scale_column_idx = [i for i in range(data_shape)]
        else:
            scale_column_idx = [i for i in range(data_shape)]

        return scale_column_idx

    def __check_equal(self, size1, size2):
        if size1 != size2:
            raise ValueError("Check equal failed, {} != {}".format(size1, size2))

    def __get_min_max_value_by_normal(self, data):
        data_shape = self._get_data_shape(data)
        self.summary_obj = MultivariateStatisticalSummary(data, -1)
        header = data.schema.get("header")

        column_min_value = self.summary_obj.get_min()
        column_min_value = [column_min_value[key] for key in header]

        column_max_value = self.summary_obj.get_max()
        column_max_value = [column_max_value[key] for key in header]

        scale_column_idx = self._get_scale_column_idx(data)

        if self.feat_upper is not None:
            if isinstance(self.feat_upper, list):
                self.__check_equal(data_shape, len(self.feat_upper))
                for i in range(data_shape):
                    if i in scale_column_idx:
                        if column_max_value[i] > self.feat_upper[i]:
                            column_max_value[i] = self.feat_upper[i]
                        if column_min_value[i] > self.feat_upper[i]:
                            column_min_value[i] = self.feat_upper[i]
            else:
                for i in range(data_shape):
                    if i in scale_column_idx:
                        if column_max_value[i] > self.feat_upper:
                            column_max_value[i] = self.feat_upper
                        if column_min_value[i] > self.feat_upper:
                            column_min_value[i] = self.feat_upper

        if self.feat_lower is not None:
            if isinstance(self.feat_lower, list):
                self.__check_equal(data_shape, len(self.feat_lower))
                for i in range(data_shape):
                    if i in scale_column_idx:
                        if column_min_value[i] < self.feat_lower[i]:
                            column_min_value[i] = self.feat_lower[i]
                        if column_max_value[i] < self.feat_lower[i]:
                            column_max_value[i] = self.feat_lower[i]
            else:
                for i in range(data_shape):
                    if i in scale_column_idx:
                        if column_min_value[i] < self.feat_lower:
                            column_min_value[i] = self.feat_lower
                        if column_max_value[i] < self.feat_lower:
                            column_max_value[i] = self.feat_lower

        return column_min_value, column_max_value

    def __get_min_max_value_by_cap(self, data):
        data_shape = self._get_data_shape(data)
        self.summary_obj = MultivariateStatisticalSummary(data, -1)
        header = data.schema.get("header")

        if self.feat_upper is None:
            self.feat_upper = 1.0

        if self.feat_lower is None:
            self.feat_lower = 0

        if self.feat_upper < self.feat_lower:
            raise ValueError("feat_upper should not less than feat_lower")

        column_min_value = self.summary_obj.get_quantile_point(self.feat_lower)
        column_min_value = [column_min_value[key] for key in header]

        column_max_value = self.summary_obj.get_quantile_point(self.feat_upper)
        column_max_value = [column_max_value[key] for key in header]

        self.__check_equal(data_shape, len(column_min_value))
        self.__check_equal(data_shape, len(column_max_value))

        return column_min_value, column_max_value

    def _get_min_max_value(self, data):
        """
        Get each column minimum and maximum
        """
        if self.mode == consts.NORMAL:
            return self.__get_min_max_value_by_normal(data)
        elif self.mode == consts.CAP:
            return self.__get_min_max_value_by_cap(data)
        else:
            raise ValueError("unknown mode of {}".format(self.mode))

    def set_column_range(self, upper, lower):
        self.column_max_value = upper
        self.column_min_value = lower

    @staticmethod
    def reset_feature_range(data, column_max_value, column_min_value, scale_column_idx):
        _data = copy.deepcopy(data)
        for i in scale_column_idx:
            value = _data.features[i]
            if value > column_max_value[i]:
                _data.features[i] = column_max_value[i]
            elif value < column_min_value[i]:
                _data.features[i] = column_min_value[i]

        return _data

    def fit_feature_range(self, data):
        if self.feat_lower is not None or self.feat_upper is not None:
            LOGGER.info("Need fit feature range")
            if not isinstance(self.column_min_value, Iterable) or not isinstance(self.column_max_value, Iterable):
                LOGGER.info(
                    "column_min_value type is:{}, column_min_value type is:{} , should be iterable, start to get new one".format(
                        type(self.column_min_value), type(self.column_max_value)))
                self.column_min_value, self.column_max_value = self._get_min_max_value(data)

            if not self.scale_column_idx:
                self.scale_column_idx = self._get_scale_column_idx(data)
                LOGGER.info("scale_column_idx is None, start to get new one, new scale_column_idx:{}".format(
                    self.scale_column_idx))

            f = functools.partial(self.reset_feature_range, column_max_value=self.column_max_value,
                                  column_min_value=self.column_min_value, scale_column_idx=self.scale_column_idx)
            fit_data = data.mapValues(f)
            fit_data.schema = data.schema

            return fit_data

        else:
            LOGGER.info("feat_lower is None and feat_upper is None, do not need to fit feature range!")
            return data

    def get_model_summary(self):
        cols_info = self._get_param().col_scale_param
        return {
        col_name: {"column_upper": col.column_upper, "column_lower": col.column_lower, "mean": col.mean, "std": col.std} for
        col_name, col in cols_info.items()}

    def export_model(self, need_run):
        meta_obj = self._get_meta(need_run)
        param_obj = self._get_param()
        result = {
            self.model_meta_name: meta_obj,
            self.model_param_name: param_obj
        }
        return result

    def fit(self, data):
        pass

    def transform(self, data):
        pass

    def load_model(self, name, namespace):
        pass

    def save_model(self, name, namespace):
        pass

    def _get_param(self):
        pass

    def _get_meta(self, need_run):
        pass
