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

from federatedml.model_base import ModelBase
from federatedml.param.column_expand_param import ColumnExpandParam
from federatedml.protobuf.generated import column_expand_meta_pb2, column_expand_param_pb2
from federatedml.util import consts, LOGGER

DELIMITER = ","


class FeatureGenerator(object):
    def __init__(self, method, append_header, fill_value):
        self.method = method
        self.append_header = append_header
        self.fill_value = fill_value
        self.append_value = self._get_append_value()
        self.generator = self._get_generator()

    def _get_append_value(self):
        if len(self.fill_value) == 0:
            return
        if len(self.fill_value) == 1:
            fill_value = str(self.fill_value[0])
            new_features = [fill_value] * len(self.append_header)
            append_value = DELIMITER.join(new_features)
        else:
            append_value = DELIMITER.join([str(v) for v in self.fill_value])
        return append_value

    def _get_generator(self):
        while True:
            yield self.append_value

    def generate(self):
        return next(self.generator)


class ColumnExpand(ModelBase):
    def __init__(self):
        super(ColumnExpand, self).__init__()
        self.model_param = ColumnExpandParam()
        self.need_run = None
        self.append_header = None
        self.method = None
        self.fill_value = None

        self.summary_obj = None
        self.header = None
        self.new_feature_generator = None

        self.model_param_name = 'ColumnExpandParam'
        self.model_meta_name = 'ColumnExpandMeta'

    def _init_model(self, params):
        self.model_param = params
        self.need_run = params.need_run
        self.append_header = params.append_header
        self.method = params.method
        self.fill_value = params.fill_value
        self.new_feature_generator = FeatureGenerator(params.method,
                                                      params.append_header,
                                                      params.fill_value)

    @staticmethod
    def _append_feature(entry, append_value):
        # empty content
        if len(entry) == 0:
            new_entry = append_value
        else:
            new_entry = entry + DELIMITER + append_value
        return new_entry

    def _append_column(self, data):
        # uses for FATE v1.5.x
        append_value = self.new_feature_generator.generate()
        new_data = data.mapValues(lambda v: ColumnExpand._append_feature(v, append_value))

        new_schema = copy.deepcopy(data.schema)
        header = new_schema.get("header", "")
        if len(header) == 0:
            new_header = DELIMITER.join(self.append_header)
            if new_schema.get("sid", None) is not None:
                new_schema["sid"] = new_schema.get("sid").strip()
        else:
            new_header = DELIMITER.join(header.split(DELIMITER) + self.append_header)
        new_schema["header"] = new_header
        new_data.schema = new_schema
        LOGGER.debug(f"new_data schema: {new_schema}")

        return new_data, new_header

    def _get_meta(self):
        meta = column_expand_meta_pb2.ColumnExpandMeta(
            append_header=self.append_header,
            method=self.method,
            fill_value=[str(v) for v in self.fill_value],
            need_run=self.need_run
        )
        return meta

    def _get_param(self):
        param = column_expand_param_pb2.ColumnExpandParam(header=self.header)
        return param

    def export_model(self):
        meta_obj = self._get_meta()
        param_obj = self._get_param()
        result = {
            self.model_meta_name: meta_obj,
            self.model_param_name: param_obj
        }
        self.model_output = result
        return result

    def load_model(self, model_dict):
        meta_obj = list(model_dict.get('model').values())[0].get(self.model_meta_name)
        param_obj = list(model_dict.get('model').values())[0].get(self.model_param_name)

        self.append_header = list(meta_obj.append_header)
        self.method = meta_obj.method
        self.fill_value = list(meta_obj.fill_value)
        self.need_run = meta_obj.need_run

        self.new_feature_generator = FeatureGenerator(self.method,
                                                      self.append_header,
                                                      self.fill_value)
        self.header = param_obj.header
        return

    def fit(self, data):
        LOGGER.info(f"Enter Column Expand fit")
        # return original value if no append header provided
        if self.method == consts.MANUAL and len(self.append_header) == 0:
            LOGGER.info(f"Finish Column Expand fit. Original data returned.")
            self.header = data.schema["header"]
            return data
        new_data, self.header = self._append_column(data)
        LOGGER.info(f"Finish Column Expand fit")
        return new_data

    def transform(self, data):
        LOGGER.info(f"Enter Column Expand transform")
        if self.method == consts.MANUAL and len(self.append_header) == 0:
            LOGGER.info(f"Finish Column Expand transform. Original data returned.")
            return data
        new_data, self.header = self._append_column(data)
        LOGGER.info(f"Finish Column Expand transform")
        return new_data
