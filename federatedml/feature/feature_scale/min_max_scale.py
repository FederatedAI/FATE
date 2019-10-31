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

import functools
import numpy as np

from federatedml.protobuf.generated.feature_scale_meta_pb2 import ScaleMeta
from federatedml.protobuf.generated.feature_scale_param_pb2 import ScaleParam
from federatedml.protobuf.generated.feature_scale_param_pb2 import ColumnScaleParam
from arch.api.utils import log_utils
from federatedml.feature.feature_scale.base_scale import BaseScale

LOGGER = log_utils.getLogger()


class MinMaxScale(BaseScale):
    """
    Transforms features by scaling each feature to a given range,e.g.between minimum and maximum. The transformation is given by:
            X_scale = (X - X.min) / (X.max - X.min), while X.min is the minimum value of feature, and X.max is the maximum
    """

    def __init__(self, params):
        super().__init__(params)
        self.mode = params.mode

        self.column_range = None

    @staticmethod
    def __scale(data, max_value_list, min_value_list, scale_value_list, process_cols_list):
        """
        Scale operator for each column. The input data type is data_instance
        """
        for i in process_cols_list:
            value = data.features[i]
            if value > max_value_list[i]:
                value = max_value_list[i]
            elif value < min_value_list[i]:
                value = min_value_list[i]

            data.features[i] = np.around((value - min_value_list[i]) / scale_value_list[i], 6)

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
        """
        self.column_min_value, self.column_max_value = self._get_min_max_value(data)
        self.scale_column_idx = self._get_scale_column_idx(data)
        self.header = self._get_header(data)

        self.column_range = []
        for i in range(len(self.column_max_value)):
            scale = self.column_max_value[i] - self.column_min_value[i]
            if scale < 0:
                raise ValueError("scale value should large than 0")
            elif np.abs(scale - 0) < 1e-6:
                scale = 1
            self.column_range.append(scale)

        f = functools.partial(MinMaxScale.__scale, max_value_list=self.column_max_value,
                              min_value_list=self.column_min_value, scale_value_list=self.column_range,
                              process_cols_list=self.scale_column_idx)
        fit_data = data.mapValues(f)

        return fit_data

    def transform(self, data):
        """
        Transform input data using min-max scale with fit results
        Parameters
        ----------
        data: data_instance, input data
        Returns
        ----------
        transform_data:data_instance, data after transform
        """

        self.column_range = []
        for i in range(len(self.column_max_value)):
            scale = self.column_max_value[i] - self.column_min_value[i]
            if scale < 0:
                raise ValueError("scale value should large than 0")
            elif np.abs(scale - 0) < 1e-6:
                scale = 1
            self.column_range.append(scale)

        f = functools.partial(MinMaxScale.__scale, max_value_list=self.column_max_value,
                              min_value_list=self.column_min_value, scale_value_list=self.column_range,
                              process_cols_list=self.scale_column_idx)

        transform_data = data.mapValues(f)

        return transform_data

    def __get_meta(self, need_run):
        if self.header:
            scale_column = [self.header[i] for i in self.scale_column_idx]
        else:
            scale_column = ["_".join(["col", str(i)]) for i in self.scale_column_idx]

        if not self.data_shape:
            self.data_shape = -1

        meta_proto_obj = ScaleMeta(method="min_max_scale",
                                   mode=self.mode,
                                   area=self.area,
                                   scale_column=scale_column,
                                   feat_upper=self._get_upper(self.data_shape),
                                   feat_lower=self._get_lower(self.data_shape),
                                   need_run=need_run
                                   )
        return meta_proto_obj

    def __get_param(self):
        min_max_scale_param_dict = {}
        if self.header:
            for i, header in enumerate(self.header):
                if i in self.scale_column_idx:
                    param_obj = ColumnScaleParam(column_upper=np.round(self.column_max_value[i], self.round_num),
                                                 column_lower=np.round(self.column_min_value[i], self.round_num))
                    min_max_scale_param_dict[header] = param_obj

        param_proto_obj = ScaleParam(col_scale_param=min_max_scale_param_dict,
                                     header=self.header)
        return param_proto_obj

    def export_model(self, need_run):
        meta_obj = self.__get_meta(need_run)
        param_obj = self.__get_param()
        result = {
            self.model_meta_name: meta_obj,
            self.model_param_name: param_obj
        }
        return result
