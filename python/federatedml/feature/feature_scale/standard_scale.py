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

import numpy as np

from federatedml.protobuf.generated.feature_scale_meta_pb2 import ScaleMeta
from federatedml.protobuf.generated.feature_scale_param_pb2 import ScaleParam
from federatedml.protobuf.generated.feature_scale_param_pb2 import ColumnScaleParam
from federatedml.feature.feature_scale.base_scale import BaseScale
from federatedml.statistic.statics import MultivariateStatisticalSummary
from federatedml.util import LOGGER


class StandardScale(BaseScale):
    """
    Standardize features by removing the mean and scaling to unit variance. The standard score of a sample x is calculated as:
    z = (x - u) / s, where u is the mean of the training samples, and s is the standard deviation of the training samples
    """

    def __init__(self, params):
        super().__init__(params)
        self.with_mean = params.with_mean
        self.with_std = params.with_std

        self.mean = None
        self.std = None

    def set_param(self, mean, std):
        self.mean = mean
        self.std = std

    @staticmethod
    def __scale_with_column_range(data, column_upper, column_lower, mean, std, process_cols_list):
        features = np.array(data.features, dtype=float)
        for i in process_cols_list:
            value = data.features[i]
            if value > column_upper[i]:
                value = column_upper[i]
            elif value < column_lower[i]:
                value = column_lower[i]

            features[i] = (value - mean[i]) / std[i]

        _data = copy.deepcopy(data)
        _data.features = features

        return _data

    @staticmethod
    def __scale(data, mean, std, process_cols_list):
        features = np.array(data.features, dtype=float)
        for i in process_cols_list:
            features[i] = (data.features[i] - mean[i]) / std[i]

        _data = copy.deepcopy(data)
        _data.features = features
        return _data

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
        self.column_min_value, self.column_max_value = self._get_min_max_value(data)
        self.scale_column_idx = self._get_scale_column_idx(data)
        self.header = self._get_header(data)
        self.data_shape = self._get_data_shape(data)

        # fit column value if larger than parameter upper or less than parameter lower
        data = self.fit_feature_range(data)

        if not self.with_mean and not self.with_std:
            self.mean = [0 for _ in range(self.data_shape)]
            self.std = [1 for _ in range(self.data_shape)]
        else:
            self.summary_obj = MultivariateStatisticalSummary(data, -1)

            if self.with_mean:
                self.mean = self.summary_obj.get_mean()
                self.mean = [self.mean[key] for key in self.header]
            else:
                self.mean = [0 for _ in range(self.data_shape)]

            if self.with_std:
                self.std = self.summary_obj.get_std_variance()
                self.std = [self.std[key] for key in self.header]

                for i, value in enumerate(self.std):
                    if np.abs(value - 0) < 1e-6:
                        self.std[i] = 1
            else:
                self.std = [1 for _ in range(self.data_shape)]

        f = functools.partial(self.__scale, mean=self.mean, std=self.std, process_cols_list=self.scale_column_idx)
        fit_data = data.mapValues(f)

        return fit_data

    def transform(self, data):
        """
        Transform input data using standard scale with fit results
        Parameters
        ----------
        data: data_instance, input data

        Returns
        ----------
        transform_data:data_instance, data after transform
        """
        f = functools.partial(self.__scale_with_column_range, column_upper=self.column_max_value,
                              column_lower=self.column_min_value,
                              mean=self.mean, std=self.std, process_cols_list=self.scale_column_idx)
        transform_data = data.mapValues(f)

        return transform_data

    def _get_meta(self, need_run):
        if self.header:
            scale_column = [self.header[i] for i in self.scale_column_idx]
        else:
            scale_column = ["_".join(["col", str(i)]) for i in self.scale_column_idx]

        if not self.data_shape:
            self.data_shape = -1

        meta_proto_obj = ScaleMeta(method="standard_scale",
                                   area="null",
                                   scale_column=scale_column,
                                   feat_upper=self._get_upper(self.data_shape),
                                   feat_lower=self._get_lower(self.data_shape),
                                   with_mean=self.with_mean,
                                   with_std=self.with_std,
                                   need_run=need_run
                                   )
        return meta_proto_obj

    def _get_param(self):
        column_scale_param_dict = {}
        if self.header:
            for i, header in enumerate(self.header):
                if i in self.scale_column_idx:
                    param_obj = ColumnScaleParam(column_upper=self.column_max_value[i],
                                                 column_lower=self.column_min_value[i],
                                                 mean=self.mean[i],
                                                 std=self.std[i])
                    column_scale_param_dict[header] = param_obj

        param_proto_obj = ScaleParam(col_scale_param=column_scale_param_dict,
                                     header=self.header)
        return param_proto_obj
