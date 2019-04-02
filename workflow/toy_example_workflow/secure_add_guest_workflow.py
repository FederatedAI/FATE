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
from federatedml.toy_example.secure_add_guest import SecureAddGuest
from workflow import status_tracer_decorator
import argparse
import json
import numpy as np
import sys
import time


class SecureAddGuestWorkflow(object):
    def __init__(self):
        self.secure_add_guest_inst = None

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
        kvs = [(i, i) for i in range(1000)]
        data_x = eggroll.parallelize(kvs, include_key=True)
        self.secure_add_guest_inst = SecureAddGuest(data_x)

    @status_tracer_decorator.status_trace
    def run(self):
        self.init_running_env()
        self.init_data()

        n = 1000
        expected_sum = n * (n - 1) // 2 + (n - 1) * n * (2 * n - 1) // 6
        secure_sum = None
        start_time = time.time()
        try:
            print ("Running...")
            secure_sum = self.secure_add_guest_inst.run()
        finally:
            end_time = time.time()
            print ("Finish, time cost is %.4f" % (end_time - start_time))

            if secure_sum is None or np.abs(expected_sum - secure_sum) > 1e-6:
                print ("Secure Add Example Task Is FAIL!!!")
            else:
                print ("Secure ADD Example Task Is OK!!!")

    
if __name__ == "__main__":
    secure_add_guest_workflow = SecureAddGuestWorkflow()
    secure_add_guest_workflow.run()
