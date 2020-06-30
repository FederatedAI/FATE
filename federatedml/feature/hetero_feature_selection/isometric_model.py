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


class SingleFeatureValues(object):
    def __init__(self, values, col_names, party_id=-1):
        self.values = values
        self.col_names = col_names
        self.party_id = party_id


class IsometricModel(object):
    """
    Use to Store Metric values

    Properties
    ----------
    feature_values: dict
        key is value name such as iv, mean etc.
        value is SingleFeatureValues obj.

    host_values: dict
        key is host_id which is a party_id string
        value is a dict like feature_values

    """
    def __init__(self):
        self.feature_values = {}
        self.host_values = {}

    def set_model_pb(self, model_pb):
        self_values = model_pb.self_values
        self.feature_values = self._parse_result_pb(self_values)

        for host_id, host_model_obj in dict(model_pb.host_values).items():
            self.host_values[host_id] = self._parse_result_pb(host_model_obj)

    @staticmethod
    def _parse_result_pb(value_obj):
        result = {}
        for value_obj in list(value_obj.results):
            value_name = value_obj.value_name
            values = list(value_obj.values)
            col_names = list(value_obj.col_names)
            if len(values) != len(col_names):
                raise ValueError(f"The length of values are not equal to the length"
                                 f" of col_names with value_name: {value_name}")
            result[value_name] = SingleFeatureValues(values, col_names)
        return result

    @property
    def valid_value_name(self):
        return list(self.feature_values.keys())