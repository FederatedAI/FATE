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
#

from arch.api import eggroll
from arch.api import federation
from federatedml.toy_example.secure_add_host import SecureAddHost
from workflow import status_tracer_decorator
import argparse
import json
import sys
import time


class SecureAddHostWorkflow(object):
    def __init__(self):
        self.secure_add_host_inst = None

    def init_eggroll(self, job_id, conf):
        eggroll.init(job_id, conf["WorkFlowParam"]["work_mode"])

    def init_federation(self, job_id, conf):
        federation.init(job_id, conf)

    def init_running_env(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--config', required=True, type=str, help="Specify a config json file path")
        parser.add_argument('-j', '--job_id', type=str, required=True, help="Specify the job id")
        args = parser.parse_args()
        config_path = args.config
        if not args.config:
            LOGGER.error("Config File should be provided")
            exit(-100)
        job_id = args.job_id
       
        fin = open(config_path, "r")
        conf = json.loads(fin.read())

        self.init_eggroll(job_id, conf)
        self.init_federation(job_id, conf)

    def init_data(self):
        kvs = [(i, i * i) for i in range(1000)]
        data_y = eggroll.parallelize(kvs, include_key=True)
        self.secure_add_host_inst = SecureAddHost(data_y)

    @status_tracer_decorator.status_trace
    def run(self):
        self.init_running_env()
        self.init_data()
        self.secure_add_host_inst.run()

    
if __name__ == "__main__":
    secure_add_host_workflow = SecureAddHostWorkflow()
    
    secure_add_host_workflow.run()
