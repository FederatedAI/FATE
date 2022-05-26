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
import math
import operator

from federatedml.feature.fate_element_type import NoneType
from federatedml.feature.feature_selection.filter_base import BaseFilterMethod
from federatedml.param.feature_selection_param import PercentageValueParam
from federatedml.protobuf.generated import feature_selection_meta_pb2
from federatedml.statistic.data_overview import is_sparse_data


class PercentageValueFilter(BaseFilterMethod):
    """
    filter the columns if all values in this feature is the same

    """

    def __init__(self, filter_param: PercentageValueParam):
        super().__init__(filter_param)

    def _parse_filter_param(self, filter_param):
        self.upper_pct = filter_param.upper_pct

    def fit(self, data_instances, suffix):

        k = 1
        while (1 / k) > self.upper_pct:
            k += 1

        total_num = data_instances.count()
        thres_num = math.ceil(total_num * self.upper_pct)
        mode_res = self._find_kth_mode(data_instances, k)
        for col_index, mode_info in mode_res.items():
            col_name = self.selection_properties.header[col_index]
            if mode_info is None:
                self.selection_properties.add_left_col_name(col_name)
                self.selection_properties.add_feature_value(col_name, False)
                continue
            mode_num = mode_info[1]
            if mode_num <= thres_num:
                self.selection_properties.add_left_col_name(col_name)
                self.selection_properties.add_feature_value(col_name, False)
            else:
                self.selection_properties.add_feature_value(col_name, True)

        self._keep_one_feature(pick_high=True)
        return self

    def _find_kth_mode(self, data_instances, k):
        """
        Find 1/k mode. If there is a mode that number of which is larger than 1/k of total nums, return this mode and
        its percentage. If there is not, return None, None.

        Parameters
        ----------
        data_instances: Table
            Original data

        k: int
        """
        is_sparse = is_sparse_data(data_instances)

        def find_mode_candidate(instances, select_cols):
            """
            Find at most k - 1 mode candidates.
            Parameters
            ----------
            instances: Data generator
                Original data
            k: int

            select_cols: list
                Indicates columns that need to be operated.

            is_sparse: bool
                Whether input data format is sparse

            Returns
            -------
            all_candidates: dict
                Each key is col_index and value is a list that contains mode candidates.
            """
            all_candidates = {}
            for col_index in select_cols:
                all_candidates[col_index] = {}

            for _, instant in instances:
                for col_index in select_cols:
                    candidate_dict = all_candidates[col_index]
                    if is_sparse:
                        feature_value = instant.features.get_data(col_index, 0)
                    else:
                        feature_value = instant.features[col_index]
                    if isinstance(feature_value, float):
                        feature_value = round(feature_value, 8)

                    if feature_value in candidate_dict:
                        candidate_dict[feature_value] += 1
                    elif len(candidate_dict) < k - 1:
                        candidate_dict[feature_value] = 1
                    else:
                        to_delete_col = []
                        for key in candidate_dict:
                            candidate_dict[key] -= 1
                            if candidate_dict[key] == 0:
                                to_delete_col.append(key)
                        for d_k in to_delete_col:
                            del candidate_dict[d_k]
            for col_index, candidate_dict in all_candidates.items():
                candidate_dict = {key: 0 for key, _ in candidate_dict.items()}
                all_candidates[col_index] = candidate_dict

            return all_candidates

        def merge_mode_candidate(d1, d2):
            assert len(d1) == len(d2)
            for col_idx, d in d1.items():
                d.update(d2[col_idx])
            return d1

        def merge_candidates_num(candi_1, candi_2):
            assert len(candi_1) == len(candi_2)
            for col_idx, candidate_dict in candi_1.items():
                candi_dict_2 = candi_2[col_idx]
                for feature_value, num in candi_dict_2.items():
                    if feature_value in candidate_dict:
                        candidate_dict[feature_value] += num
                    else:
                        candidate_dict[feature_value] = num
            return candi_1

        def static_candidates_num(instances, select_cols, all_candidates):
            """
            Static number of candidates
            Parameters
            ----------
            instances: Data generator
                Original data

            select_cols: list
                Indicates columns that need to be operated.

            all_candidates: dict
                Each key is col_index and value is a list that contains mode candidates.
            """

            for _, instant in instances:
                for col_index in select_cols:
                    candidate_dict = all_candidates[col_index]
                    if is_sparse:
                        feature_value = instant.features.get_data(col_index, NoneType())
                    else:
                        feature_value = instant.features[col_index]
                    if isinstance(feature_value, float):
                        feature_value = round(feature_value, 8)

                    if feature_value in candidate_dict:
                        candidate_dict[feature_value] += 1

            # mode_result = {}
            # for col_index, candidate_dict in all_candidates.items():
            #     feature_value, nums = sorted(candidate_dict.items(), key=operator.itemgetter(1), reverse=False)[0]
            #     mode_result[col_index] = (feature_value, nums)
            return all_candidates

        find_func = functools.partial(find_mode_candidate,
                                      select_cols=self.selection_properties.select_col_indexes)
        all_candidates = data_instances.applyPartitions(find_func).reduce(merge_mode_candidate)
        static_func = functools.partial(static_candidates_num,
                                        select_cols=self.selection_properties.select_col_indexes,
                                        all_candidates=all_candidates)
        mode_candidate_statics = data_instances.applyPartitions(static_func).reduce(merge_candidates_num)
        result = {}
        for col_index, candidate_dict in mode_candidate_statics.items():
            if len(candidate_dict) > 0:
                res = sorted(candidate_dict.items(), key=operator.itemgetter(1), reverse=True)[0]
            else:
                res = None
            result[col_index] = res

        return result

    # def get_meta_obj(self, meta_dicts):
    #     result = feature_selection_meta_pb2.PercentageValueFilterMeta(upper_pct=self.upper_pct)
    #     meta_dicts['pencentage_value_meta'] = result
    #     return meta_dicts

    def get_meta_obj(self):
        result = feature_selection_meta_pb2.FilterMeta()
        return result
