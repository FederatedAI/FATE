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
import argparse
import sys
import traceback
import json


ROLE = 'guest'
PARTY_ID = 9999
COMPONENT_NAME = 'hetero_feature_selection_0'

# GUEST_FEATURE_NAMES = []
GUEST_FEATURE_NAMES = -1
HOST_FEATURE_INDICES = [[1, 2, 3]]

# Support 'iv', 'woe_array', 'is_woe_monotonic', 'split_points'
RESULT_LIST = ['iv', 'woe_array', 'is_woe_monotonic']

DESCENDING = True
WRITE_RESULT = False


def decode_col_name(encoded_name: str):
    try:
        col_index = int(encoded_name.split('.')[1])
    except IndexError or ValueError:
        raise RuntimeError("Trying to decode an invalid col_name.")
    return col_index


class QuerySelectionResult(base_task.BaseTask):
    def __init__(self):
        super().__init__()
        self.feature_objs = []
        self.host_results = []
        self.host_party_ids = []

    def query_cpn_result(self, job_id, role, party_id, cpn):

        cmd = ['python', run_config.FATE_FLOW_PATH, "-f", "component_output_model", "-j", job_id,
               '-cpn', cpn, '-r', role, '-p', str(party_id)]
        result_json = self.start_task(cmd)
        # self.write_json_file(result_json, run_config.TEMP_DATA_PATH + '/selection_result.json')
        return result_json

    def parse_result(self, result_json):
        pass

    def generated_results(self):
        result = {}
        sort_result = sorted(self.feature_objs, key=lambda obj: obj.iv, reverse=DESCENDING)
        for result_name in RESULT_LIST:
            result_key = '_'.join([result_name, 'result'])
            result[result_key] = [(obj.name, getattr(obj, result_name)) for obj in sort_result]

        for host_idx, host_result in enumerate(self.host_results):
            party_id = self.host_party_ids[host_idx]
            host_result = {}
            host_sorted_result = sorted(self.host_results[host_idx], key=lambda obj: obj.iv, reverse=DESCENDING)
            for result_name in RESULT_LIST:
                result_key = '_'.join([result_name, 'result'])
                host_result[result_key] = [(obj.name, getattr(obj, result_name)) for obj in host_sorted_result]
            result['_'.join(['host', party_id])] = host_result

        print("Result is {}".format(result))
        if WRITE_RESULT:
            self.write_json_file(result, run_config.TEMP_DATA_PATH + 'feature_binning_results.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument('-c', '--conf', required=False, type=str, help="input conf path", default=None)
    parser.add_argument('-cpn', '--component_name', required=False, type=str, help="component name",
                        default=COMPONENT_NAME)
    parser.add_argument('-r', '--role', required=False, type=str, help="role", default=ROLE)
    parser.add_argument('-p', '--party_id', required=False, type=str, help="party_id", default=PARTY_ID)
    parser.add_argument('-j', '--job_id', required=True, type=str, help="job id")

    try:
        args = parser.parse_args()
        task_obj = QuerySelectionResult()
        result_json = task_obj.query_cpn_result(args.job_id, args.role, args.party_id, args.component_name)

        # print("query bin result: {}".format(result_json))
        # task_obj.parse_result(result_json)
        # task_obj.generated_results()

    except Exception as e:
        exc_type, exc_value, exc_traceback_obj = sys.exc_info()
        response = {'retcode': 100, 'retmsg': traceback.format_exception(exc_type, exc_value, exc_traceback_obj)}
        print(json.dumps(response, indent=4))
        print()
