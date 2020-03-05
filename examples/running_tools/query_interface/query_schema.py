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

from examples.running_tools.base_task import BaseTask
from examples.running_tools import run_config
import argparse
import sys
import traceback
import json


ROLE = 'host'
PARTY_ID = 10000
feature_idx = -1
# feature_idx = [0, 1, 2]


class QuerySchema(BaseTask):

    def query_component_output_data(self, job_id, cpn, this_feature_idx=None):
        if this_feature_idx is None:
            this_feature_idx = feature_idx
        cmd = ['python', run_config.FATE_FLOW_PATH, "-f", "component_output_data", "-j", job_id,
               '-cpn', cpn, '-r', ROLE, '-p', str(PARTY_ID), '-o', run_config.TEMP_DATA_PATH, '-l', '10']
        stdout = self.start_task(cmd)
        print("query_component_output_data result: {}".format(stdout))
        job_folder_name = 'job_{}_{}_{}_{}_output_data'.format(job_id, cpn, ROLE, PARTY_ID)
        file_name = run_config.TEMP_DATA_PATH + job_folder_name + '/output_data_meta.json'
        meta_json = self.read_json_file(file_name)
        header = meta_json.get('header')
        result = []
        for idx, header_name in enumerate(header[1:]):
            if this_feature_idx == -1 or idx in this_feature_idx:
                result.append((idx, header_name))
        print("Queried header is {}".format(result))
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument('-c', '--conf', required=False, type=str, help="input conf path", default=None)
    parser.add_argument('-cpn', '--component_name', required=False, type=str, help="component name",
                        default='dataio_0')
    parser.add_argument('-j', '--job_id', required=True, type=str, help="job id")

    try:
        args = parser.parse_args()
        task_obj = QuerySchema()
        task_obj.query_component_output_data(str(args.job_id), args.component_name)

    except Exception as e:
        exc_type, exc_value, exc_traceback_obj = sys.exc_info()
        response = {'retcode': 100, 'retmsg': traceback.format_exception(exc_type, exc_value, exc_traceback_obj)}
        print(json.dumps(response, indent=4))
        print()
