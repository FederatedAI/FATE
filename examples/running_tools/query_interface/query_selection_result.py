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

SHOW_DETAIL_FILTERS = True

WRITE_RESULT = False


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
        final_left_cols = [x for x, _ in result_json['data']['finalLeftCols']['leftCols'].items()]
        left_col_nums = len(final_left_cols)
        result = {'final_left_cols': final_left_cols,
                  "left_col_nums": left_col_nums}
        if SHOW_DETAIL_FILTERS:
            filter_results = result_json['data']['results']
            detail_results = {}
            for f_r in filter_results:
                _left_cols = [x for x, _ in f_r['leftCols']['leftCols'].items()]
                detail_results[f_r['filterName']] = {
                    "left_cols": _left_cols,
                    "left_cols_nums": len(_left_cols)
                }
            result['filter_results'] = detail_results
        print("Result is {}".format(result))
        if WRITE_RESULT:
            self.write_json_file(result, run_config.TEMP_DATA_PATH + 'feature_selection_results.json')


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
        task_obj.parse_result(result_json)
        # task_obj.generated_results()

    except Exception as e:
        exc_type, exc_value, exc_traceback_obj = sys.exc_info()
        response = {'retcode': 100, 'retmsg': traceback.format_exception(exc_type, exc_value, exc_traceback_obj)}
        print(json.dumps(response, indent=4))
        print()
