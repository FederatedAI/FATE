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
from federatedml.statistic import data_overview


def empty_table_detection(data_instances):
    num_data = data_instances.count()
    if num_data == 0:
        raise ValueError(f"Count of data_instance is 0: {data_instances}")


def empty_feature_detection(data_instances):
    is_empty_feature = data_overview.is_empty_feature(data_instances)
    if is_empty_feature:
        raise ValueError(f"Number of features of DTable is 0: {data_instances}")


def sparse_vector_column_gathering(iterable, is_sparse=True):

    appeared_columns = set()
    for k, v in iterable:
        if is_sparse:
            feature_dict = v.features.sparse_vec
            appeared_columns.update(feature_dict.keys())
        else:
            features = v.features
            appeared_columns.update(np.where(~np.isnan(features))[0])

    return appeared_columns


def merge_column_sets(v1: set, v2: set):
    return v1.update(v2)


def empty_column_detection(data_instance):

    contains_empty_columns = False
    lost_feat = []
    is_sparse = data_overview.is_sparse_data(data_instance)
    map_func = functools.partial(sparse_vector_column_gathering, is_sparse=is_sparse)
    map_rs = data_instance.mapPartitions(map_func)
    reduce_rs = map_rs.reduce(merge_column_sets)

    if not is_sparse:
        # select col name by index and transform it to set
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
    if schema is None:
        return
    header = schema.get("header", None)
    if header is not None:
        for col_name in header:
            if not col_name.isprintable():
                raise ValueError(f"non-printable char found in header column {col_name}, please check.")
        header_set = set(header)
        if len(header_set) != len(header):
            raise ValueError(f"data header contains repeated values, please check.")

    sid_name = schema.get("sid_name", None)
    if sid_name is not None and not sid_name.isprintable():
        raise ValueError(f"non-printable char found in sid_name {sid_name}, please check.")

    label_name = schema.get("label_name", None)
    if label_name is not None and not label_name.isprintable():
        raise ValueError(f"non-printable char found in label_name {label_name}, please check.")
