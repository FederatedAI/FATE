#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
#  Copyright 2021 The FATE Authors. All Rights Reserved.
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

import copy
import numpy as np

from federatedml.model_base import Metric, MetricMeta
from federatedml.model_base import ModelBase
from federatedml.param.label_transform_param import LabelTransformParam
from federatedml.protobuf.generated import label_transform_meta_pb2, label_transform_param_pb2
from federatedml.statistic.data_overview import get_label_count, get_predict_result_labels
from federatedml.util import LOGGER


class LabelTransformer(ModelBase):
    def __init__(self):
        super().__init__()
        self.model_param = LabelTransformParam()
        self.metric_name = "label_transform"
        self.metric_namespace = "train"
        self.metric_type = "LABEL_TRANSFORM"
        self.model_param_name = 'LabelTransformParam'
        self.model_meta_name = 'LabelTransformMeta'
        self.weight_mode = None
        self.encoder_key_type = None
        self.encoder_value_type = None
        self.label_encoder = None
        self.label_list = None

    def _init_model(self, params):
        self.model_param = params
        self.label_encoder = params.label_encoder
        self.label_list = params.label_list
        self.need_run = params.need_run

    def update_label_encoder(self, data):
        if self.label_encoder is not None:
            LOGGER.info(f"label encoder provided")
            if self.label_list is not None:
                LOGGER.info(f"label list provided")
                self.encoder_key_type = {str(v): type(v).__name__ for v in self.label_list}

        else:
            data_type = data.schema.get("content_type")
            if data_type is None:
                label_count = get_label_count(data)
                labels = sorted(label_count.keys())
            # predict result
            else:
                labels = sorted(get_predict_result_labels(data))
            self.label_encoder = dict(zip(labels, range(len(labels))))

        if self.encoder_key_type is None:
            self.encoder_key_type = {str(k): type(k).__name__ for k in self.label_encoder.keys()}
        self.encoder_value_type = {str(k): type(v).__name__ for k, v in self.label_encoder.items()}

        self.label_encoder = {load_value_to_type(k,
                                                 self.encoder_key_type[str(k)]): v for k,
                              v in self.label_encoder.items()}

    def _get_meta(self):
        meta = label_transform_meta_pb2.LabelTransformMeta(
            need_run=self.need_run
        )
        return meta

    def _get_param(self):
        label_encoder = self.label_encoder
        if self.label_encoder is not None:
            label_encoder = {str(k): str(v) for k, v in self.label_encoder.items()}
        param = label_transform_param_pb2.LabelTransformParam(
            label_encoder=label_encoder,
            encoder_key_type=self.encoder_key_type,
            encoder_value_type=self.encoder_value_type)
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

        self.need_run = meta_obj.need_run

        self.encoder_key_type = param_obj.encoder_key_type
        self.encoder_value_type = param_obj.encoder_value_type
        self.label_encoder = {
            load_value_to_type(k, self.encoder_key_type[k]): load_value_to_type(v, self.encoder_value_type[k])
            for k, v in param_obj.label_encoder.items()
        }

        return

    def callback_info(self):
        metric_meta = MetricMeta(name='train',
                                 metric_type=self.metric_type,
                                 extra_metas={
                                     "label_encoder": self.label_encoder
                                 })

        self.callback_metric(metric_name=self.metric_name,
                             metric_namespace=self.metric_namespace,
                             metric_data=[Metric(self.metric_name, 0)])
        self.tracker.set_metric_meta(metric_namespace=self.metric_namespace,
                                     metric_name=self.metric_name,
                                     metric_meta=metric_meta)

    @staticmethod
    def replace_instance_label(instance, label_encoder):
        new_instance = copy.deepcopy(instance)
        label_replace_val = label_encoder.get(instance.label)
        if label_replace_val is None:
            raise ValueError(f"{instance.label} not found in given label encoder")
        new_instance.label = label_replace_val
        return new_instance

    @staticmethod
    def replace_predict_label(predict_inst, label_encoder):
        transform_predict_inst = copy.deepcopy(predict_inst)
        true_label, predict_label, predict_score, predict_detail, result_type = transform_predict_inst.features
        # true_label, predict_label = label_encoder[true_label], label_encoder[predict_label]
        true_label_replace_val, predict_label_replace_val = label_encoder.get(
            true_label), label_encoder.get(predict_label)
        if true_label_replace_val is None:
            raise ValueError(f"{true_label_replace_val} not found in given label encoder")
        if predict_label_replace_val is None:
            raise ValueError(f"{predict_label_replace_val} not found in given label encoder")
        label_encoder_detail = {str(k): v for k, v in label_encoder.items()}
        predict_detail = {label_encoder_detail[label]: score for label, score in predict_detail.items()}
        transform_predict_inst.features = [true_label_replace_val, predict_label_replace_val, predict_score,
                                           predict_detail, result_type]
        return transform_predict_inst

    @staticmethod
    def replace_predict_label_cluster(predict_inst, label_encoder):
        transform_predict_inst = copy.deepcopy(predict_inst)
        true_label, predict_label = transform_predict_inst.features[0], transform_predict_inst.features[1]
        true_label, predict_label = label_encoder[true_label], label_encoder[predict_label]
        transform_predict_inst.features = [true_label, predict_label]
        return transform_predict_inst

    @staticmethod
    def transform_data_label(data, label_encoder):
        data_type = data.schema.get("content_type")
        if data_type == "cluster_result":
            return data.mapValues(lambda v: LabelTransformer.replace_predict_label_cluster(v, label_encoder))
        elif data_type == "predict_result":
            predict_detail = data.first()[1].features[3]
            if predict_detail == 1 and list(predict_detail.keys())[0] == "label":
                LOGGER.info(f"Regression prediction result provided. Original data returned.")
                return data
            return data.mapValues(lambda v: LabelTransformer.replace_predict_label(v, label_encoder))
        elif data_type is None:
            return data.mapValues(lambda v: LabelTransformer.replace_instance_label(v, label_encoder))
        else:
            raise ValueError(f"unknown data type: {data_type} encountered. Label transform aborted.")

    def transform(self, data):
        LOGGER.info(f"Enter Label Transformer Transform")
        if self.label_encoder is None:
            raise ValueError(f"Input Label Encoder is None. Label Transform aborted.")

        label_encoder = self.label_encoder
        data_type = data.schema.get("content_type")
        # revert label encoding if predict result
        if data_type is not None:
            label_encoder = dict(zip(self.label_encoder.values(), self.label_encoder.keys()))

        result_data = LabelTransformer.transform_data_label(data, label_encoder)
        result_data.schema = data.schema
        self.callback_info()

        return result_data

    def fit(self, data):
        LOGGER.info(f"Enter Label Transform Fit")
        self.update_label_encoder(data)

        result_data = LabelTransformer.transform_data_label(data, self.label_encoder)
        result_data.schema = data.schema
        self.callback_info()

        return result_data

# also used in feature imputation, to be moved to common util


def load_value_to_type(value, value_type):
    if value is None:
        loaded_value = None
    elif value_type in ["int", "int64", "long", "float", "float64", "double"]:
        loaded_value = getattr(np, value_type)(value)
    elif value_type in ["str", "_str"]:
        loaded_value = str(value)
    else:
        raise ValueError(f"unknown value type: {value_type}")
    return loaded_value
