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

import argparse
import json
import sys
import time
import traceback

import pandas as pd

from examples.running_tools import run_config
from examples.running_tools.base_task import BaseTask


HAS_EVAL = False

class ModelTask(BaseTask):

    def make_dsl_file(self, component_list):
        component_history = {}
        dsl_json = {}
        last_module = 'args'
        for cpn in component_list:
            component_history.setdefault(cpn, 0)
            cpn_name = cpn + str(component_history.get(cpn))
            module_name = self.__get_module_name(cpn)
            dsl_json[cpn_name] = {
                "module": module_name,
                "input": {
                    "data": {

                    }
                }
            }
            component_history[cpn] += 1

        dsl_json = {'components': dsl_json}


    def __get_module_name(self, cpn):
        for module_name in run_config.ALL_MODULES:
            lower_module_name = module_name.lower()
            if lower_module_name == cpn.replace('_', ''):
                return module_name
        raise ValueError("Cannot recognize component name: {}".format(cpn))





if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--func', required=False, type=str, help="role",
                        choices=('upload', 'check', 'destroy'), default='upload'
                        )

    try:
        args = parser.parse_args()
        data_task_obj = ModelTask()

    except Exception as e:
        exc_type, exc_value, exc_traceback_obj = sys.exc_info()
        response = {'retcode': 100, 'retmsg': traceback.format_exception(exc_type, exc_value, exc_traceback_obj)}
        print(json.dumps(response, indent=4))
        print()