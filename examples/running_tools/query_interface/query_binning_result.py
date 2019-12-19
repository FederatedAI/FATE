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

from examples.running_tools.query_interface import query_schema
from examples.running_tools import base_task
from examples.running_tools import run_config


ROLE = 'guest'
PARTY_ID = 9999
COMPONENT_NAME = 'hetero_feature_binning_0'

GUEST_FEATURE_NAMES = []
HOST_FEATURE_INDICES = []

# Support 'iv', 'woeArray', 'isWoeMonotonic', 'splitPoints'
RESULT_LIST = ['iv', 'woeArray', 'isWoeMonotonic']


class FeatureResult(object):
    def __init__(self):
        self.name = None
        self.iv = None
        self.woe_array = None
        self.is_woe_monotonic = False

    def _parse_(self):
        pass


class QueryBinningResult(base_task.BaseTask):

    def query_binning_result(self, job_id, role, party_id, cpn):

        cmd = ['python', run_config.FATE_FLOW_PATH, "-f", "component_output_model", "-j", job_id,
               '-cpn', cpn, '-r', role, '-p', str(party_id)]
        result_json = self.start_task(cmd)
        return result_json

    def parse_result(self, result_json):
        bin_results = result_json['data']['binningResult']['binningResult']
        result = {}
        feature_objs = []
        for feature_name, feature_result in bin_results.items():
            pass


    def _parse_iv(self):
        pass