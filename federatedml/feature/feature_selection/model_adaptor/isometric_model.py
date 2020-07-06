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
import copy


class SingleMetricInfo(object):
    """
    Use to Store Metric values

    Parameters
    ----------
    values: ndarray or list
        List of metric value of each column. Do not accept missing value.

    col_names: list
        List of column_names of above list whose length should match with above values.

    host_party_ids: list of int (party_id, such as 9999)
        If it is a federated metric, list of host party ids

    host_values: list of ndarray
        The outer list specify each host's values. The inner list are the values of
        this party

    host_col_names: list of list
        Similar to host_values where the content is col_names

    """

    def __init__(self, values, col_names, host_party_ids=None,
                 host_values=None, host_col_names=None):

        if host_party_ids is None:
            host_party_ids = []
        if host_values is None:
            host_values = []
        if host_col_names is None:
            host_col_names = []

        self.values = values
        self.col_names = col_names
        self.host_party_ids = host_party_ids
        self.host_values = host_values
        self.host_col_names = host_col_names

        self.check()

    def check(self):
        if len(self.values) != len(self.col_names):
            raise ValueError("When creating SingleMetricValue, length of values "
                             "and length of col_names should be equal")

        if not (len(self.host_party_ids) == len(self.host_values) == len(self.host_col_names)):
            raise ValueError("When creating SingleMetricValue, length of values "
                             "and length of col_names and host_party_ids should be equal")

    def union_result(self):
        values = list(self.values)
        col_names = [("guest", x) for x in self.col_names]

        for idx, host_id in enumerate(self.host_party_ids):
            values.extend(self.host_values[idx])
            col_names.extend([(host_id, x) for x in self.host_col_names[idx]])

        if len(values) != len(col_names):
            raise AssertionError("union values and col_names should have same length")
        values = np.array(values)
        return values, col_names

    def get_values(self):
        return copy.deepcopy(self.values)

    def get_col_names(self):
        return copy.deepcopy(self.col_names)



class IsometricModel(object):
    """
    Use to Store Metric values

    Parameters
    ----------
    metric_name: list of str
        The metric name, eg. iv. If a single string

    metric_info: list of SingleMetricInfo


    """

    def __init__(self, metric_name=None, metric_info=None):
        if metric_name is None:
            metric_name = []

        if not isinstance(metric_name, list):
            metric_name = [metric_name]

        if metric_info is None:
            metric_info = []

        if not isinstance(metric_info, list):
            metric_info = [metric_info]

        self._metric_names = metric_name
        self._metric_info = metric_info

    # def set_model_pb(self, model_pb):
    #     self_values = model_pb.self_values
    #     self.feature_values = self._parse_result_pb(self_values)
    #
    #     for host_id, host_model_obj in dict(model_pb.host_values).items():
    #         self.host_values[host_id] = self._parse_result_pb(host_model_obj)

    def add_metric_value(self, metric_name, metric_info):
        self._metric_names.append(metric_name)
        self._metric_info.append(metric_info)

    # @staticmethod
    # def _parse_result_pb(value_obj):
    #     result = {}
    #     for value_obj in list(value_obj.results):
    #         value_name = value_obj.value_name
    #         values = list(value_obj.values)
    #         col_names = list(value_obj.col_names)
    #         if len(values) != len(col_names):
    #             raise ValueError(f"The length of values are not equal to the length"
    #                              f" of col_names with value_name: {value_name}")
    #         result[value_name] = SingleMetricValues(values, col_names)
    #     return result

    @property
    def valid_value_name(self):
        return self._metric_names

    def get_metric_info(self, metric_name):
        if metric_name not in self.valid_value_name:
            return None
        return self._metric_info[self._metric_names.index(metric_name)]
