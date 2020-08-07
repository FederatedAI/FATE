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

import functools

from arch.api.utils import log_utils
from federatedml.util import consts

LOGGER = log_utils.getLogger()


def get_features_shape(data_instances):
    one_feature = data_instances.first()
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

def header_alignment(data_instances, pre_header):
    header = data_instances.schema["header"]
    if len((set(header) & set(pre_header))) != len(pre_header):
        raise ValueError("fit & transform data' header should be same")

    if pre_header == header:
        return data_instances

    if len(pre_header) != len(header):
        LOGGER.warning(
            "header in prediction stage is super-set training stage, predict size is {}, training header size is {}".format(
                len(header), len(pre_header)))
    else:
        LOGGER.warning("header in prediction stage will be shuffled to match the header of training stage")

    header_idx_mapping = dict(zip(pre_header, [i for i in range(len(pre_header))]))
    header_correct = {}
    for i in range(len(header)):
        col = header[i]
        if col not in header_idx_mapping:
            continue
        header_correct[i] = header_idx_mapping[col]


    def align_header(inst, header_pos=None):
        if type(inst.features).__name__ == consts.SPARSE_VECTOR:
            shape = len(header_pos)
            new_data = {}
            for k, v in inst.features.get_all_data():
                if k not in header_pos:
                    continue
                new_data[header_pos.get(k)] = v

            inst.features.set_shape(shape)
            inst.features.set_sparse_vector(new_data)
        else:
            col_order = [None] * len(header_pos)
            for k, v in header_pos.items():
                col_order[v] = k
            inst.features = inst.features[col_order]

        return inst

    data_instances = data_instances.mapValues(lambda inst: align_header(inst, header_pos=header_correct))

    return data_instances


def get_data_shape(data):
    one_feature = data.first()
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


def count_labels(data_instance):
    def _count_labels(instances):
        labels = set()
        for idx, instance in instances:
            label = instance.label
            labels.add(label)
        return labels

    label_set = data_instance.mapPartitions(_count_labels)
    label_set = label_set.reduce(lambda x1, x2: x1.union(x2))
    return len(label_set)
    # if len(label_set) != 2:
    #     return False
    # return True


def rubbish_clear(rubbish_list):
    """
    Temporary procession for resource recovery. This will be discarded in next version because of our new resource recovery plan
    Parameter
    ----------
    rubbish_list: list of DTable, each DTable in this will be destroy
    """
    for r in rubbish_list:
        try:
            if r is None:
                continue
            r.destroy()
        except Exception as e:
            LOGGER.warning("destroy Dtable error,:{}, but this can be ignored sometimes".format(e))


class DataStatistics(object):
    def __init__(self):
        self.multivariate_statistic_obj = None

    def static_all_values(self, data_instances, static_col_indexes, is_sparse: bool = False):
        if not is_sparse:
            f = functools.partial(self.__dense_values_set,
                                  static_col_indexes=static_col_indexes)
        else:
            f = functools.partial(self.__sparse_values_set,
                                  static_col_indexes=static_col_indexes)
        result_sets = data_instances.mapPartitions(f).reduce(self.__reduce_set_results)
        result = [sorted(list(x)) for x in result_sets]
        return result

    @staticmethod
    def __dense_values_set(instances, static_col_indexes: list):
        result = [set() for _ in static_col_indexes]
        for _, instance in instances:
            for idx, col_index in enumerate(static_col_indexes):
                value_set = result[idx]
                value_set.add(instance.features[col_index])
        return result

    @staticmethod
    def __sparse_values_set(instances, static_col_indexes: list):
        tmp_result = {idx: set() for idx in static_col_indexes}
        for _, instance in instances:
            data_generator = instance.features.get_all_data()
            for idx, value in data_generator:
                if idx not in tmp_result:
                    continue
                tmp_result[idx].add(value)
        result = [tmp_result[x] for x in static_col_indexes]
        return result

    @staticmethod
    def __reduce_set_results(result_set_a, result_set_b):
        final_result_sets = []
        for set_a, set_b in zip(result_set_a, result_set_b):
            final_result_sets.append(set_a.union(set_b))
        return final_result_sets
