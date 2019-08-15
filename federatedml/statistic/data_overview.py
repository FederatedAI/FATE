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

import types

from arch.api.utils import log_utils
from federatedml.util import consts

LOGGER = log_utils.getLogger()


def get_features_shape(data_instances):
    if not isinstance(data_instances, types.GeneratorType):
        features = data_instances.collect()
    else:
        features = data_instances

    try:
        one_feature = features.__next__()
    except StopIteration:
        one_feature = None

    instance = one_feature[1]
    if instance is None:
        return None

    if one_feature is not None:
        if type(one_feature[1].features).__name__ == consts.SPARSE_VECTOR:
            return one_feature[1].features.get_shape()
        else:
            return one_feature[1].features.shape[0]
    else:
        return None


def get_data_shape(data):
    if not isinstance(data, types.GeneratorType):
        features = data.collect()
    else:
        features = data

    try:
        one_feature = features.__next__()
    except StopIteration:
        one_feature = None

    if one_feature is not None:
        return len(list(one_feature[1]))
    else:
        return None


def get_header(data_instances):
    header = data_instances.schema.get('header')  # ['x1', 'x2', 'x3' ... ]
    return header


def is_empty_feature(data_instances):
    shape_of_feature = get_features_shape(data_instances)
    if shape_of_feature is None or shape_of_feature == 0:
        return True
    return False


def is_sparse_data(data_instance):
    first_data = data_instance.first()
    if type(first_data[1]).__name__ in ['ndarray', 'list']:
        return False

    data_feature = first_data[1].features
    if type(data_feature).__name__ == "ndarray":
        return False
    else:
        return True


def is_binary_labels(data_instance):
    def count_labels(instances):
        labels = set()
        for idx, instance in instances:
            label = instance.label
            labels.add(label)
        return labels

    label_set = data_instance.mapPartitions(count_labels)
    label_set = label_set.reduce(lambda x1, x2: x1.union(x2))
    if len(label_set) != 2:
        return False
    return True


def rubbish_clear(rubbish_list):
    """
    Temporary procession for resource recovery. This will be discarded in next version because of our new resource recovery plan
    Parameter
    ----------
    rubbish_list: list of DTable, each DTable in this will be destroy
    """
    for r in rubbish_list:
        try:
            r.destroy()
        except Exception as e:
            LOGGER.warning("destroy Dtable error,:{}, but this can be ignored sometimes".format(e))
