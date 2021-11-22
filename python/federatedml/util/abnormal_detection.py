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

import numpy as np
import functools
import copy
from federatedml.statistic import data_overview
from federatedml.util import LOGGER


def empty_table_detection(data_instances):
    num_data = data_instances.count()
    if num_data == 0:
        raise ValueError(f"Count of data_instance is 0: {data_instances}")


def empty_feature_detection(data_instances):
    is_empty_feature = data_overview.is_empty_feature(data_instances)
    if is_empty_feature:
        raise ValueError(f"Number of features of Table is 0: {data_instances}")


def column_gathering(iterable, ):

    lost_columns = set()
    for k, v in iterable:
        features = v.features
        lost_columns.update(np.where(~np.isnan(features))[0])

    return lost_columns


def merge_column_sets(v1: set, v2: set):
    v1_copy = copy.deepcopy(v1)
    v2_copy = copy.deepcopy(v2)
    v1_copy.update(v2_copy)
    return v1_copy


def empty_column_detection(data_instance):

    contains_empty_columns = False
    lost_feat = []
    is_sparse = data_overview.is_sparse_data(data_instance)
    if is_sparse:
        raise ValueError('sparse format empty column detection is not supported for now')
    map_func = functools.partial(column_gathering, )
    map_rs = data_instance.applyPartitions(map_func)
    reduce_rs = map_rs.reduce(merge_column_sets)

    # transform col index to col name
    reduce_rs = np.array(data_instance.schema['header'])[list(reduce_rs)]
    reduce_rs = set(reduce_rs)

    if reduce_rs != set(data_instance.schema['header']):
        lost_feat = list(set(data_instance.schema['header']).difference(reduce_rs))
        contains_empty_columns = True

    if contains_empty_columns:
        raise ValueError('column(s) {} contain(s) no values'.format(lost_feat))


def check_legal_schema(schema):
    # check for repeated header & illegal/non-printable chars except for space
    # allow non-ascii chars
    LOGGER.debug(f"schema is {schema}")
    if schema is None:
        return
    header = schema.get("header", None)
    LOGGER.debug(f"header is {header}")
    if header is not None:
        for col_name in header:
            if not col_name.isprintable():
                raise ValueError(f"non-printable char found in header column {col_name}, please check.")
        header_set = set(header)
        if len(header_set) != len(header):
            raise ValueError(f"data header contains repeated names, please check.")

    sid_name = schema.get("sid_name", None)
    LOGGER.debug(f"sid_name is {sid_name}")
    if sid_name is not None and not sid_name.isprintable():
        raise ValueError(f"non-printable char found in sid_name {sid_name}, please check.")

    label_name = schema.get("label_name", None)
    LOGGER.debug(f"label_name is {label_name}")
    if label_name is not None and not label_name.isprintable():
        raise ValueError(f"non-printable char found in label_name {label_name}, please check.")
