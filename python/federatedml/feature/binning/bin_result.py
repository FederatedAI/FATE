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

from federatedml.protobuf.generated import feature_binning_param_pb2
from federatedml.util import LOGGER


class BinColResults(object):
    def __init__(self, woe_array=(), iv_array=(), event_count_array=(), non_event_count_array=(),
                 event_rate_array=(), non_event_rate_array=(), iv=None, optimal_metric_array=()):
        self.woe_array = list(woe_array)
        self.iv_array = list(iv_array)
        self.event_count_array = list(event_count_array)
        self.non_event_count_array = list(non_event_count_array)
        self.event_rate_array = list(event_rate_array)
        self.non_event_rate_array = list(non_event_rate_array)
        self.split_points = None
        if iv is None:
            iv = 0
            for idx, woe in enumerate(self.woe_array):
                non_event_rate = non_event_count_array[idx]
                event_rate = event_rate_array[idx]
                iv += (non_event_rate - event_rate) * woe
        self.iv = iv
        self._bin_anonymous = None
        self.optimal_metric_array = list(optimal_metric_array)

    @property
    def bin_anonymous(self):
        if self.split_points is None or len(self.split_points) == 0:
            return []
        if self._bin_anonymous is None:
            return ["bin_" + str(i) for i in range(len(self.split_points))]
        return self._bin_anonymous

    @bin_anonymous.setter
    def bin_anonymous(self, x):
        self._bin_anonymous = x

    def set_split_points(self, split_points):
        self.split_points = split_points

    def set_optimal_metric(self, metric_array):
        self.optimal_metric_array = metric_array

    def get_split_points(self):
        return np.array(self.split_points)

    @property
    def is_woe_monotonic(self):
        """
        Check the woe is monotonic or not
        """
        woe_array = self.woe_array
        if len(woe_array) <= 1:
            return True

        is_increasing = all(x <= y for x, y in zip(woe_array, woe_array[1:]))
        is_decreasing = all(x >= y for x, y in zip(woe_array, woe_array[1:]))
        return is_increasing or is_decreasing

    @property
    def bin_nums(self):
        return len(self.woe_array)

    def result_dict(self):
        save_dict = self.__dict__
        save_dict['is_woe_monotonic'] = self.is_woe_monotonic
        save_dict['bin_nums'] = self.bin_nums
        return save_dict

    def reconstruct(self, iv_obj):
        self.woe_array = list(iv_obj.woe_array)
        self.iv_array = list(iv_obj.iv_array)
        self.event_count_array = list(iv_obj.event_count_array)
        self.non_event_count_array = list(iv_obj.non_event_count_array)
        self.event_rate_array = list(iv_obj.event_rate_array)
        self.non_event_rate_array = list(iv_obj.non_event_rate_array)
        self.split_points = list(iv_obj.split_points)
        self.iv = iv_obj.iv
        # new attribute since ver 1.10
        if hasattr(iv_obj, "optimal_metric_array"):
            self.optimal_metric_array = list(iv_obj.optimal_metric_array)

    def generate_pb_dict(self):
        result = {
            "woe_array": self.woe_array,
            "iv_array": self.iv_array,
            "event_count_array": self.event_count_array,
            "non_event_count_array": self.non_event_count_array,
            "event_rate_array": self.event_rate_array,
            "non_event_rate_array": self.non_event_rate_array,
            "split_points": self.split_points,
            "iv": self.iv,
            "is_woe_monotonic": self.is_woe_monotonic,
            "bin_nums": self.bin_nums,
            "bin_anonymous": self.bin_anonymous,
            "optimal_metric_array": self.optimal_metric_array
        }
        return result


class SplitPointsResult(object):
    def __init__(self):
        self.split_results = {}
        self.optimal_metric = {}

    def put_col_split_points(self, col_name, split_points):
        self.split_results[col_name] = split_points

    def put_col_optimal_metric_array(self, col_name, metric_array):
        self.optimal_metric[col_name] = metric_array

    @property
    def all_split_points(self):
        return self.split_results

    @property
    def all_optimal_metric(self):
        return self.optimal_metric

    def get_split_points_array(self, col_names):
        split_points_result = []
        for col_name in col_names:
            if col_name not in self.split_results:
                continue
            split_points_result.append(self.split_results[col_name])
        return np.array(split_points_result)

    def to_json(self):
        return {k: list(v) for k, v in self.split_results.items()}


class BinResults(object):
    def __init__(self):
        self.all_cols_results = {}  # {col_name: BinColResult}
        self.role = ''
        self.party_id = ''

    def set_role_party(self, role, party_id):
        self.role = role
        self.party_id = party_id

    def put_col_results(self, col_name, col_results: BinColResults):
        ori_col_results = self.all_cols_results.get(col_name)
        if ori_col_results is not None:
            col_results.set_split_points(ori_col_results.get_split_points())
        self.all_cols_results[col_name] = col_results

    def put_col_split_points(self, col_name, split_points):
        col_results = self.all_cols_results.get(col_name, BinColResults())
        col_results.set_split_points(split_points)
        self.all_cols_results[col_name] = col_results

    def query_split_points(self, col_name):
        col_results = self.all_cols_results.get(col_name)
        if col_results is None:
            LOGGER.warning("Querying non-exist split_points")
            return None
        return col_results.split_points

    def put_optimal_metric_array(self, col_name, metric_array):
        col_results = self.all_cols_results.get(col_name, BinColResults())
        col_results.set_optimal_metric(metric_array)
        self.all_cols_results[col_name] = col_results

    @property
    def all_split_points(self):
        results = {}
        for col_name, col_result in self.all_cols_results.items():
            results[col_name] = col_result.get_split_points()
        return results

    @property
    def all_ivs(self):
        return [(col_name, x.iv) for col_name, x in self.all_cols_results.items()]

    @property
    def all_woes(self):
        return {col_name: x.woe_array for col_name, x in self.all_cols_results.items()}

    @property
    def all_monotonic(self):
        return {col_name: x.is_woe_monotonic for col_name, x in self.all_cols_results.items()}

    @property
    def all_optimal_metric(self):
        return {col_name: x.optimal_metric_array for col_name, x in self.all_cols_results.items()}

    def summary(self, split_points=None):
        if split_points is None:
            split_points = {}
            for col_name, x in self.all_cols_results.items():
                sp = x.get_split_points().tolist()
                split_points[col_name] = sp
        # split_points = {col_name: x.split_points for col_name, x in self.all_cols_results.items()}
        return {"iv": self.all_ivs,
                "woe": self.all_woes,
                "monotonic": self.all_monotonic,
                "split_points": split_points}

    def generated_pb(self, split_points=None):
        col_result_dict = {}
        if split_points is not None:
            for col_name, sp in split_points.items():
                self.put_col_split_points(col_name, sp)
        for col_name, col_bin_result in self.all_cols_results.items():
            bin_res_dict = col_bin_result.generate_pb_dict()
            # LOGGER.debug(f"col name: {col_name}, bin_res_dict: {bin_res_dict}")
            col_result_dict[col_name] = feature_binning_param_pb2.IVParam(**bin_res_dict)
        # LOGGER.debug("In generated_pb, role: {}, party_id: {}".format(self.role, self.party_id))
        result_pb = feature_binning_param_pb2.FeatureBinningResult(binning_result=col_result_dict,
                                                                   role=self.role,
                                                                   party_id=str(self.party_id))
        return result_pb

    def reconstruct(self, result_pb):
        self.role = result_pb.role
        self.party_id = result_pb.party_id
        binning_result = dict(result_pb.binning_result)
        for col_name, col_bin_result in binning_result.items():
            col_bin_obj = BinColResults()
            col_bin_obj.reconstruct(col_bin_result)
            self.all_cols_results[col_name] = col_bin_obj
        return self

    def update_anonymous(self, anonymous_header_dict):
        all_cols_results = dict()
        for col_name, col_bin_result in self.all_cols_results.items():
            updated_col_name = anonymous_header_dict[col_name]
            all_cols_results[updated_col_name] = col_bin_result

        self.all_cols_results = all_cols_results
        return self


class MultiClassBinResult(BinResults):
    def __init__(self, labels):
        super().__init__()
        self.labels = labels
        if len(self.labels) == 2:
            self.is_multi_class = False
            self.bin_results = [BinResults()]
        else:
            self.is_multi_class = True
            self.bin_results = [BinResults() for _ in range(len(self.labels))]

    def set_role_party(self, role, party_id):
        self.role = role
        self.party_id = party_id
        for br in self.bin_results:
            br.set_role_party(role, party_id)

    def put_col_results(self, col_name, col_results: BinColResults, label_idx=0):
        self.bin_results[label_idx].put_col_results(col_name, col_results)

    def summary(self, split_points=None):
        if not self.is_multi_class:
            return {"result": self.bin_results[0].summary(split_points)}
        return {label: self.bin_results[label_idx].summary(split_points) for
                label_idx, label in enumerate(self.labels)}

    def put_col_split_points(self, col_name, split_points, label_idx=None):
        if label_idx is None:
            for br in self.bin_results:
                br.put_col_split_points(col_name, split_points)
        else:
            self.bin_results[label_idx].put_col_split_points(col_name, split_points)

    def put_optimal_metric_array(self, col_name, metric_array, label_idx=None):
        if label_idx is None:
            for br in self.bin_results:
                br.put_optimal_metric_array(col_name, metric_array)
        else:
            self.bin_results[label_idx].put_optimal_metric_array(col_name, metric_array)

    def generated_pb_list(self, split_points=None):
        res = []
        for br in self.bin_results:
            res.append(br.generated_pb(split_points))
        return res

    @staticmethod
    def reconstruct(result_pb, labels=None):
        if not isinstance(result_pb, list):
            result_pb = [result_pb]

        if labels is None:
            if len(result_pb) <= 1:
                labels = [0, 1]
            else:
                labels = list(range(len(result_pb)))
        result = MultiClassBinResult(labels)
        for idx, pb in enumerate(result_pb):
            result.bin_results[idx].reconstruct(pb)

        return result

    def update_anonymous(self, anonymous_header_dict):
        for idx in range(len(self.bin_results)):
            self.bin_results[idx].update_anonymous(anonymous_header_dict)

    @property
    def all_split_points(self):
        return self.bin_results[0].all_split_points
