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

from federatedml.util import consts, LOGGER


def get_class_weight(data_instances):
    class_weight  = data_instances.mapPartitions(compute_class_weight).reduce(lambda x, y: dict(Counter(x) + Counter(y)))
    n_samples = data_instances.count()
    n_classes = len(class_weight.keys())
    class_weight.update((k, n_samples / (n_classes * v)) for k, v in class_weight.items())

    return class_weight


def compute_class_weight(kv_iterator):
    class_dict = {}
    for _, inst in kv_iterator:
        count = class_dict.get(inst.label, 0)
        class_dict[inst.label] = count + 1

    if len(class_dict.keys()) > consts.MAX_CLASSNUM:
        raise ValueError("In Classify Task, max dif classes should no more than %d" % (consts.MAX_CLASSNUM))

    return class_dict


def replace_weight(data_instance, class_weight, weight_loc=None, weight_mean=None):
    weighted_data_instance = copy.copy(data_instance)
    original_features = weighted_data_instance.features
    if weight_loc:
        weighted_data_instance.set_weight(original_features[weight_loc] / weight_mean)
        weighted_data_instance.features = original_features[np.arange(original_features) != weight_loc]
    else:
        weighted_data_instance.set_weight(class_weight.get(data_instance.label, 1))
    return weighted_data_instance


def assign_sample_weight(data_instances, class_weight, weight_loc):
    weight_mean = None
    if weight_loc:
        def sum_sample_weight(kv_iterator):
            sample_weight = 0
            for _, inst in kv_iterator:
                sample_weight += inst.features[weight_loc]
            return sample_weight
        weight_sum = data_instances.mapPartitions(sum_sample_weight).reduce(lambda x, y: x + y)
        weight_mean = weight_sum / data_instances.count()
    return data_instances.mapValues(lambda v: replace_weight(v, class_weight, weight_loc, weight_mean))


def transform_weighted_instance(data_instances, class_weight='balanced', weight_loc=None):
    if class_weight == 'balanced':
        class_weight = get_class_weight(data_instances)
    return assign_sample_weight(data_instances, class_weight, weight_loc)


def compute_weight_array(data_instances, class_weight='balanced'):
    if class_weight is None:
        class_weight = {}
    elif class_weight == 'balanced':
        class_weight = compute_class_weight(data_instances)
    weight_inst = data_instances.mapValues(lambda v: class_weight.get(v.label, 1))
    return np.array([v[1] for v in list(weight_inst.collect())])
