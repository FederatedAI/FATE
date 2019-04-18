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
from federatedml.util import consts


def get_features_shape(data_instances):
    # LOGGER.debug("In get features shape method, data_instances count: {}".format(
    #     data_instances.count()
    # ))
    if not isinstance(data_instances, types.GeneratorType):
        features = data_instances.collect()
    else:
        features = data_instances

    try:
        one_feature = features.__next__()
    except StopIteration:
        # LOGGER.warning("Data instances is Empty")
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
    # LOGGER.debug("In get features shape method, data count: {}".format(
    #     data.count()
    # ))
    if not isinstance(data, types.GeneratorType):
        features = data.collect()
    else:
        features = data

    try:
        one_feature = features.__next__()
    except StopIteration:
        # LOGGER.warning("Data instances is Empty")
        one_feature = None

    if one_feature is not None:
        return len(list(one_feature[1]))
    else:
        return None


def is_empty_feature(data_instances):
    shape_of_feature = get_features_shape(data_instances)
    if shape_of_feature is None or shape_of_feature == 0:
        return True
    return False
