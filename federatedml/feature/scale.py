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

from arch.api.utils import log_utils
from fate_flow.entity.metric import MetricMeta
from federatedml.feature.feature_scale.min_max_scale import MinMaxScale
from federatedml.feature.feature_scale.standard_scale import StandardScale
from federatedml.model_base import ModelBase
from federatedml.param.scale_param import ScaleParam
from federatedml.util import consts

LOGGER = log_utils.getLogger()

class Scale(ModelBase):
    """
    The Scale class is used to data scale. MinMaxScale and StandardScale is supported now
    """

    def __init__(self):
        super().__init__()
        self.model_name = None
        self.model_param_name = 'ScaleParam'
        self.model_meta_name = 'ScaleMeta'
        self.model_param = ScaleParam()

        self.scale_param_obj = None
        self.scale_obj = None
        self.header = None
        self.column_max_value = None
        self.column_min_value = None
        self.mean = None
        self.std = None
        self.scale_column_idx = None


    def fit(self, data):
        """
        Apply scale for input data
        Parameters
        ----------
        data: data_instance, input data

        Returns
        ----------
        data:data_instance, data after scale
        scale_value_results: list, the fit results information of scale
        """
        LOGGER.info("Start scale data fit ...")

        if self.model_param.method == consts.MINMAXSCALE:
            self.scale_obj = MinMaxScale(self.model_param)
        elif self.model_param.method == consts.STANDARDSCALE:
            self.scale_obj = StandardScale(self.model_param)
        else:
            LOGGER.warning("Scale method is {}, do nothing and return!".format(self.model_param.method))

        if self.scale_obj:
            fit_data = self.scale_obj.fit(data)
            fit_data.schema = data.schema

            self.callback_meta(metric_name="scale", metric_namespace="train",
                               metric_meta=MetricMeta(name="scale", metric_type="SCALE", extra_metas={"method":self.model_param.method}))
        else:
            fit_data = data

        LOGGER.info("End fit data ...")
        return fit_data

    def transform(self, data, fit_config=None):
        """
        Transform input data using scale with fit results
        Parameters
        ----------
        data: data_instance, input data
        fit_config: list, the fit results information of scale

        Returns
        ----------
        transform_data:data_instance, data after transform
        """
        LOGGER.info("Start scale data transform ...")

        if self.model_param.method == consts.MINMAXSCALE:
            self.scale_obj = MinMaxScale(self.model_param)
        elif self.model_param.method == consts.STANDARDSCALE:
            self.scale_obj = StandardScale(self.model_param)
            self.scale_obj.set_param(self.mean, self.std)
        else:
            LOGGER.info("DataTransform method is {}, do nothing and return!".format(self.model_param.method))

        if self.scale_obj:
            self.scale_obj.header = self.header
            self.scale_obj.scale_column_idx = self.scale_column_idx
            self.scale_obj.set_column_range(self.column_max_value, self.column_min_value)
            transform_data = self.scale_obj.transform(data)
            transform_data.schema = data.schema

            self.callback_meta(metric_name="scale", metric_namespace="train",
                               metric_meta=MetricMeta(name="scale", metric_type="SCALE",
                                                      extra_metas={"method": self.model_param.method}))
        else:
            transform_data = data

        LOGGER.info("End transform data.")

        return transform_data


    def load_model(self, model_dict):
        model_obj = list(model_dict.get('model').values())[0].get(self.model_param_name)
        meta_obj = list(model_dict.get('model').values())[0].get(self.model_meta_name)
        self.header = list(model_obj.header)
        self.need_run = meta_obj.need_run

        shape = len(self.header)
        self.column_max_value = [ 0 for _ in range(shape) ]
        self.column_min_value = [0 for _ in range(shape)]
        self.mean = [0 for _ in range(shape)]
        self.std = [1 for _ in range(shape)]
        self.scale_column_idx = []
        scale_param_dict = dict(model_obj.col_scale_param)
        for key, column_scale_param in scale_param_dict.items():
            index = self.header.index(key)
            self.scale_column_idx.append(index)

            self.column_max_value[index] = column_scale_param.column_upper
            self.column_min_value[index] = column_scale_param.column_lower
            self.mean[index] = column_scale_param.mean
            self.std[index] = column_scale_param.std

        self.scale_column_idx.sort()

    def export_model(self):
        if not self.scale_obj:
            if self.model_param.method == consts.MINMAXSCALE:
                self.scale_obj = MinMaxScale(self.model_param)
            else:
                self.scale_obj = StandardScale(self.model_param)

        return self.scale_obj.export_model(self.need_run)





