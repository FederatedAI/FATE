#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

import copy
import numpy as np
from collections import Counter

from federatedml.model_base import ModelBase
from federatedml.param.sample_weight_param import SampleWeightParam
from federatedml.protobuf.generated import sample_weight_meta_pb2, sample_weight_param_pb2
from federatedml.util import consts, LOGGER


def compute_weight_array(data_instances):
    weight_inst = data_instances.mapValues(lambda v: v.weight)
    return np.array([v[1] for v in list(weight_inst.collect())])


def compute_class_weight(kv_iterator):
    class_dict = {}
    for _, inst in kv_iterator:
        count = class_dict.get(inst.label, 0)
        class_dict[inst.label] = count + 1

    if len(class_dict.keys()) > consts.MAX_CLASSNUM:
        raise ValueError("In Classify Task, max dif classes should no more than %d" % (consts.MAX_CLASSNUM))

    return class_dict


class SampleWeight(ModelBase):
    def __init__(self):
        super().__init__()
        self.model_param = SampleWeightParam()
        self.model_name = 'SampleWeight'
        self.model_param_name = 'SampleWeightParam'
        self.model_meta_name = 'SampleWeightMeta'

    def _init_model(self, params):
        self.model_param = params
        self.class_weight = params.class_weight
        self.sample_weight_name = params.sample_weight_name
        self.normalize = params.normalize
        self.need_run = params.need_run

    @staticmethod
    def get_class_weight(data_instances):
        class_weight = data_instances.mapPartitions(compute_class_weight).reduce(
            lambda x, y: dict(Counter(x) + Counter(y)))
        n_samples = data_instances.count()
        n_classes = len(class_weight.keys())
        class_weight.update((k, n_samples / (n_classes * v)) for k, v in class_weight.items())

        return class_weight

    @staticmethod
    def replace_weight(data_instance, class_weight, weight_loc=None, weight_base=None):
        weighted_data_instance = copy.copy(data_instance)
        original_features = weighted_data_instance.features
        if weight_loc:
            weighted_data_instance.set_weight(original_features[weight_loc] / weight_base)
            weighted_data_instance.features = original_features[np.arange(original_features.shape[0]) != weight_loc]
        else:
            weighted_data_instance.set_weight(class_weight.get(data_instance.label, 1))
        return weighted_data_instance

    @staticmethod
    def assign_sample_weight(data_instances, class_weight, weight_loc, normalize):
        weight_base = 1
        if weight_loc and normalize:
            def sum_sample_weight(kv_iterator):
                sample_weight = 0
                for _, inst in kv_iterator:
                    sample_weight += inst.features[weight_loc]
                return sample_weight

            weight_sum = data_instances.mapPartitions(sum_sample_weight).reduce(lambda x, y: x + y)
            weight_base = weight_sum / data_instances.count()
        return data_instances.mapValues(lambda v: SampleWeight.replace_weight(v, class_weight, weight_loc, weight_base))

    @staticmethod
    def get_weight_loc(data_instances, sample_weight_name):
        weight_loc = None
        if sample_weight_name:
            try:
                weight_loc = data_instances.schema["header"].index(sample_weight_name)
            except ValueError:
                return
        return weight_loc

    def transform_weighted_instance(self, data_instances, weight_loc):
        if self.class_weight and self.class_weight == 'balanced':
            self.class_weight = SampleWeight.get_class_weight(data_instances)
        return SampleWeight.assign_sample_weight(data_instances, self.class_weight, weight_loc, self.normalize)

    def export_model(self):
        class_weight=None
        if self.class_weight:
            class_weight = {str(k): v for k, v in self.class_weight.items()}
        LOGGER.debug(f"class weight exported is: {class_weight}")
        meta_obj = sample_weight_meta_pb2.SampleWeightMeta(sample_weight_name=self.sample_weight_name,
                                                           need_run=self.need_run)
        param_obj = sample_weight_param_pb2.SampleWeightParam(class_weight=class_weight)
        result = {
            self.model_meta_name: meta_obj,
            self.model_param_name: param_obj
        }
        return result

    def load_model(self, model_dict):
        result_obj = list(model_dict.get('model').values())[0].get(
            self.model_param_name)

        meta_obj = list(model_dict.get('model').values())[0].get(self.model_meta_name)

        self.need_run, self.sample_weight_name = meta_obj.need_run, meta_obj.sample_weight_name
        tmp_class_weight = dict(result_obj.class_weight)
        self.class_weight = {int(k): v for k, v in tmp_class_weight.items()}

    def fit(self, data_instances):
        if self.sample_weight_name is None and self.class_weight is None:
            return data_instances

        if self.class_weight and isinstance(self.class_weight, dict):
            self.class_weight = {int(k): v for k, v in self.class_weight.items()}

        if self.sample_weight_name and self.class_weight:
            LOGGER.warning(f"Both 'sample_weight_name' and 'class_weight' provided."
                           f"Only weight from 'sample_weight_name' is used.")

        new_schema = copy.deepcopy(data_instances.schema)
        weight_loc = None
        if self.sample_weight_name:
            weight_loc = SampleWeight.get_weight_loc(data_instances, self.sample_weight_name)
            if weight_loc:
                new_schema["header"].pop(weight_loc)
            else:
                LOGGER.warning(f"Cannot find weight column of given sample_weight_name '{self.sample_weight_name}'."
                               f"Original data returned.")
                return data_instances
        result_instances = self.transform_weighted_instance(data_instances, weight_loc)
        result_instances.schema = new_schema
        return result_instances
