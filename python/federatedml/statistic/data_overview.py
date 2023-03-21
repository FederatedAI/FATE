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
import functools
import json
from collections import Counter

import numpy as np

from federatedml.feature.instance import Instance
from federatedml.util import LOGGER
from federatedml.util import consts


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


def get_instance_shape(instance):
    if instance is None:
        return None

    if type(instance.features).__name__ == consts.SPARSE_VECTOR:
        return instance.features.get_shape()
    else:
        return instance.features.shape[0]


def get_anonymous_header(data_instances):
    anonymous_header = data_instances.schema.get('anonymous_header')  # ['x1', 'x2', 'x3' ... ]
    return anonymous_header


def look_up_names_from_header(name_list, source_header, transform_header):
    """

    Parameters
    ----------
    name_list: list or str, list of feature name(s)
    source_header: table header containing name_list
    transform_header: table header into which name_list to be transformed

    Returns
    -------
    list of plaintext feature names

    """
    if name_list is None:
        return
    if len(source_header) != len(transform_header):
        raise ValueError(f"Length of source header and transform header do not match, please check.")
    if not isinstance(name_list, list):
        name_list = [name_list]
    name_set = set(name_list)
    # name list contains repeated name
    if len(name_set) < len(name_list):
        LOGGER.debug(f"Repeated name(s) found in provided name_list: {name_list}.")
        name_set = name_list

    feature_names = [f_name for i, f_name in enumerate(transform_header) if source_header[i] in name_set]
    if len(feature_names) < len(name_set):
        raise ValueError(f"Cannot match all provided names from: {name_list} to given header, "
                         f"please check.")
    return feature_names


def max_abs_sample_weight_map_func(kv_iter):
    max_weight = -1
    for k, inst in kv_iter:
        if np.abs(inst.weight) > max_weight:
            max_weight = np.abs(inst.weight)

    return max_weight


def max_sample_weight_cmp(v1, v2):
    return v1 if v1 > v2 else v2


def get_max_sample_weight(data_inst_with_weight):
    inter_rs = data_inst_with_weight.applyPartitions(max_abs_sample_weight_map_func)
    max_weight = inter_rs.reduce(max_sample_weight_cmp)
    return max_weight


def check_negative_sample_weight(kv_iterator):
    for k, v in kv_iterator:
        if isinstance(v, Instance) and v.weight is not None:
            if v.weight < 0:
                return True
    return False


def header_alignment(data_instances, pre_header, pre_anonymous_header=None):
    header = [col.strip() for col in data_instances.schema["header"]]
    if len((set(header) & set(pre_header))) != len(pre_header):
        raise ValueError(f"fit & transform data' header should be the same! "
                         f"Previous header: {pre_header}. "
                         f"Current header: {header}.")

    if pre_header == header:
        if pre_anonymous_header:
            data_instances.schema["anonymous_header"] = pre_anonymous_header
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
            inst_new = copy.deepcopy(inst)
            inst_new.features.set_shape(shape)
            inst_new.features.set_sparse_vector(new_data)
        else:
            col_order = [None] * len(header_pos)
            for k, v in header_pos.items():
                col_order[v] = k
            inst_new = copy.deepcopy(inst)
            inst_new.features = inst.features[col_order]

        return inst_new

    correct_schema = data_instances.schema
    correct_schema["header"] = pre_header
    if pre_anonymous_header:
        correct_schema["anonymous_header"] = pre_anonymous_header
    data_instances = data_instances.mapValues(lambda inst: align_header(inst, header_pos=header_correct))
    data_instances.schema = correct_schema
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
    if type(first_data[1]).__name__ in ['ndarray', 'list', 'tuple']:
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

    label_set = data_instance.applyPartitions(_count_labels)
    label_set = label_set.reduce(lambda x1, x2: x1.union(x2))
    return len(label_set)
    # if len(label_set) != 2:
    #     return False
    # return True


def with_weight(data_instances):
    first_entry = data_instances.first()[1]
    if isinstance(first_entry, Instance) and first_entry.weight is not None:
        return True
    return False


def get_class_dict(kv_iterator):
    class_dict = {}
    for _, inst in kv_iterator:
        count = class_dict.get(inst.label, 0)
        class_dict[inst.label] = count + 1

    if len(class_dict.keys()) > consts.MAX_CLASSNUM:
        raise ValueError("In Classify Task, max dif classes should be no more than %d" % (consts.MAX_CLASSNUM))

    return class_dict


def get_label_count(data_instances):
    class_weight = data_instances.mapPartitions(get_class_dict).reduce(
        lambda x, y: dict(Counter(x) + Counter(y)))
    return class_weight


def get_predict_result_labels(data):
    def _get_labels(score_inst):
        labels = set()
        for idx, result in score_inst:
            true_label = result.features[0]
            predict_label = result.features[1]
            labels.add(true_label)
            labels.add(predict_label)
        return labels

    label_set = data.applyPartitions(_get_labels)
    label_set = label_set.reduce(lambda x1, x2: x1.union(x2))
    if len(label_set) > consts.MAX_CLASSNUM:
        raise ValueError("In Classify Task, max dif classes should be no more than %d" % (consts.MAX_CLASSNUM))

    return label_set


def rubbish_clear(rubbish_list):
    """
    Temporary procession for resource recovery. This will be discarded in next version because of our new resource recovery plan
    Parameter
    ----------
    rubbish_list: list of Table, each Table in this will be destroy
    """
    for r in rubbish_list:
        try:
            if r is None:
                continue
            r.destroy()
        except Exception as e:
            LOGGER.warning("destroy table error,:{}, but this can be ignored sometimes".format(e))


def check_with_inst_id(data_instances):
    instance = data_instances.first()[1]
    if isinstance(instance, Instance) and instance.with_inst_id:
        return True
    return False


def predict_detail_dict_to_str(result_dict):
    return "\"" + json.dumps(result_dict).replace("\"", "\'") + "\""


def predict_detail_str_to_dict(result_dict_str):
    return json.loads(json.loads(result_dict_str).replace("\'", "\""))


def scale_sample_weight(data_instances):
    data_count = data_instances.count()

    def _sum_all_weight(kv_iterator):
        weight_sum = 0
        for _, v in kv_iterator:
            weight_sum += v.weight
        return weight_sum

    total_weight = data_instances.mapPartitions(_sum_all_weight).reduce(lambda x, y: x + y)
    # LOGGER.debug(f"weight_sum is : {total_weight}")
    scale_factor = data_count / total_weight

    # LOGGER.debug(f"scale factor is : {total_weight}")

    def _replace_weight(instance):
        new_weight = instance.weight * scale_factor
        instance.set_weight(new_weight)
        return instance

    scaled_instances = data_instances.mapValues(lambda v: _replace_weight(v))
    return scaled_instances


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
        result_sets = data_instances.applyPartitions(f).reduce(self.__reduce_set_results)
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
