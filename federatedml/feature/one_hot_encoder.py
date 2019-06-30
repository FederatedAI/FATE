#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import functools

import numpy as np

from arch.api.proto import onehot_meta_pb2, onehot_param_pb2
from arch.api.utils import log_utils
from federatedml.model_base import ModelBase
from federatedml.statistic.data_overview import get_header
from federatedml.util import consts
from federatedml.param.onehot_encoder_param import OneHotEncoderParam

LOGGER = log_utils.getLogger()

MODEL_PARAM_NAME = 'OneHotParam'
MODEL_META_NAME = 'OneHotMeta'
MODEL_NAME = 'OneHotEncoder'


class OneHotEncoder(ModelBase):
    def __init__(self):
        super(OneHotEncoder, self).__init__()
        self.cols = []
        self.header = []
        self.col_maps = {}
        self.cols_dict = {}
        self.output_data = None
        self.model_param = OneHotEncoderParam()

    def _init_model(self, model_param):
        self.model_param = model_param
        self.cols_index = model_param.cols

    def fit(self, data_instances):
        self._init_cols(data_instances)

        f1 = functools.partial(self.record_new_header,
                               cols=self.cols,
                               cols_dict=self.cols_dict)

        col_maps = data_instances.mapPartitions(f1)

        f2 = functools.partial(self.merge_col_maps)
        col_maps = col_maps.reduce(f2)
        self._detect_overflow(col_maps)
        self.col_maps = col_maps
        self.set_schema(data_instances)
        data_instances = self.transform(data_instances)

        return data_instances

    def _detect_overflow(self, col_maps):
        for col_name, col_value_map in col_maps.items():
            if len(col_value_map) > consts.ONE_HOT_LIMIT:
                raise ValueError("Input data should not have more than {} possible value when doing one-hot encode"
                                 .format(consts.ONE_HOT_LIMIT))

    def transform(self, data_instances):
        self._init_cols(data_instances)
        ori_header = self.header.copy()
        self._transform_schema()
        f = functools.partial(self.transfer_one_instance,
                              col_maps=self.col_maps,
                              ori_header=ori_header,
                              transformed_header=self.header)
        new_data = data_instances.mapValues(f)
        self.set_schema(new_data)
        return new_data

    def _transform_schema(self):

        header = self.header
        LOGGER.info("[Result][OneHotEncoder]Before one-hot, data_instances schema is : {}".format(header))
        for col_name, value_map in self.col_maps.items():
            col_idx = header.index(col_name)
            new_headers = list(value_map.values())
            if col_idx == 0:
                header = new_headers + header[1:]
            else:
                header = header[:col_idx] + new_headers + header[col_idx + 1:]

        self.cols_dict = {}
        for col in header:
            col_index = header.index(col)
            self.cols_dict[col] = col_index
        self.header = header
        LOGGER.info("[Result][OneHotEncoder]After one-hot, data_instances schema is : {}".format(header))

    def _init_cols(self, data_instances):
        header = get_header(data_instances)
        self.header = header
        if self.cols_index == -1:
            self.cols = header
        else:
            cols = []
            for idx in self.cols_index:
                try:
                    idx = int(idx)
                except ValueError:
                    raise ValueError("In binning module, selected index: {} is not integer".format(idx))

                if idx >= len(header):
                    raise ValueError(
                        "In binning module, selected index: {} exceed length of data dimension".format(idx))
                cols.append(header[idx])
            self.cols = cols

        self.cols_dict = {}
        for col in self.cols:
            col_index = header.index(col)
            self.cols_dict[col] = col_index

    @staticmethod
    def record_new_header(data, cols, cols_dict):
        """
        Generate a new schema based on data value. Each new value will generate a new header.

        Returns
        -------
        col_maps: a dict in which keys are original header, values are dicts. The dicts in value
        """
        col_maps = {}
        for col_name in cols:
            col_maps[col_name] = {}

        for _, instance in data:
            feature = instance.features
            for col_name in cols:
                this_col_map = col_maps.get(col_name)
                col_index = cols_dict.get(col_name)
                feature_value = feature[col_index]
                feature_value = str(feature_value)
                if feature_value not in this_col_map:
                    new_feature_header = str(col_name) + '_' + str(feature_value)
                    this_col_map[feature_value] = new_feature_header

        return col_maps

    @staticmethod
    def merge_col_maps(col_map1, col_map2):
        if col_map1 is None and col_map2 is None:
            return None

        if col_map1 is None:
            return col_map2

        if col_map2 is None:
            return col_map1

        for col_name, value_dict in col_map2.items():
            if col_name not in col_map1:
                col_map1[col_name] = value_dict
                continue
            else:
                col_1_value_dict = col_map1[col_name]
                for value, header in value_dict.items():
                    if value not in col_1_value_dict:
                        col_1_value_dict[value] = header

        return col_map1

    @staticmethod
    def transfer_one_instance(instance, col_maps, ori_header, transformed_header):
        feature = instance.features
        feature_dict = {}
        for idx, col_name in enumerate(ori_header):
            feature_dict[col_name] = feature[idx]

        for col_name in transformed_header:
            if col_name not in feature_dict:
                feature_dict[col_name] = 0

        for col_name, value_dict in col_maps.items():
            feature_value = feature_dict.get(col_name)
            feature_value = str(feature_value)
            header_name = value_dict.get(feature_value)
            feature_dict[header_name] = 1

        feature_array = []
        for col_name in transformed_header:
            feature_array.append(feature_dict[col_name])

        feature_array = np.array(feature_array, dtype=float)
        instance.features = feature_array
        return instance

    def set_schema(self, data_instance):
        data_instance.schema = {"header": self.header}

    def _get_meta(self):
        meta_protobuf_obj = onehot_meta_pb2.OneHotMeta(cols=self.cols)
        return meta_protobuf_obj

    def _get_param(self):
        pb_dict = {}
        for col_name, value_dict in self.col_maps.items():
            value_dict_obj = onehot_param_pb2.ColDict(encode_map=value_dict)
            pb_dict[col_name] = value_dict_obj

        result_obj = onehot_param_pb2.OneHotParam(col_map=pb_dict)
        return result_obj

    def save_model(self):
        meta_obj = self._get_meta()
        param_obj = self._get_param()
        result = {
            MODEL_META_NAME: meta_obj,
            MODEL_PARAM_NAME: param_obj
        }
        return result

    def _load_model(self, model_dict):
        self._parse_need_run(model_dict, MODEL_META_NAME)
        model_param = list(model_dict.get('model').values())[0].get(MODEL_PARAM_NAME)
        # model_meta = model_dict.get(MODEL_NAME).get(MODEL_META_NAME)

        self.col_maps = dict(model_param.col_map)
        for k, v in self.col_maps.items():
            self.col_maps[k] = dict(v.encode_map)
