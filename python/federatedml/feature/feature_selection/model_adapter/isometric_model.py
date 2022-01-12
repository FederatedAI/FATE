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

from federatedml.util import LOGGER


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

    def get_partial_values(self, select_col_names, party_id=None):
        """
        Return values selected by provided col_names.
        Use party_id to indicate which party to get. If None, obtain from values,
        otherwise, obtain from host_values
        """
        if party_id is None:
            col_name_map = {name: idx for idx, name in enumerate(self.col_names)}
            col_indices = [col_name_map[x] for x in select_col_names]
            values = np.array(self.values)[col_indices]
        else:
            if party_id not in self.host_party_ids:
                raise ValueError(f"party_id: {party_id} is not in host_party_ids:"
                                 f" {self.host_party_ids}")
            party_idx = self.host_party_ids.index(party_id)
            col_name_map = {name: idx for idx, name in
                            enumerate(self.host_col_names[party_idx])}
            # LOGGER.debug(f"col_name_map: {col_name_map}")

            values = []
            host_values = np.array(self.host_values[party_idx])
            for host_col_name in select_col_names:
                if host_col_name in col_name_map:
                    values.append(host_values[col_name_map[host_col_name]])
                else:
                    values.append(0)

            # col_indices = [col_name_map[x] for x in select_col_names]
            # values = np.array(self.host_values[party_idx])[col_indices]
        return list(values)


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

    def add_metric_value(self, metric_name, metric_info):
        self._metric_names.append(metric_name)
        self._metric_info.append(metric_info)

    @property
    def valid_value_name(self):
        return self._metric_names

    def get_metric_info(self, metric_name):
        LOGGER.debug(f"valid_value_name: {self.valid_value_name}, "
                     f"metric_name: {metric_name}")
        if metric_name not in self.valid_value_name:
            return None
        return self._metric_info[self._metric_names.index(metric_name)]

    def get_all_metric_info(self):
        return self._metric_info
