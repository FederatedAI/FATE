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
from examples.toy_example.secure_add_host import SecureAddHost
import sys
import time


class SecureAddHostWorkflow(object):
    def __init__(self, jobid, guest_partyid, host_partyid, work_mode=0):
        self.init_eggroll(jobid, work_mode)
        self.init_federation(jobid, guest_partyid, host_partyid)

        self.secure_add_host_inst = None
        self.init_data()

    def init_eggroll(self, jobid, work_mode):
        eggroll.init(jobid, work_mode)

    def init_federation(self, jobid, guest_partyid, host_partyid):
        runtime_conf = {"local" : 
                           {"role": "host",
                            "party_id": int(host_partyid)},
                        "role":
                           {"host": [int(host_partyid)],
                            "guest": [int(guest_partyid)]}}

        federation.init(jobid, runtime_conf)

    def init_data(self):
        kvs = [(i, i * i) for i in range(1000)]
        data_y = eggroll.parallelize(kvs, include_key=True)
        self.secure_add_host_inst = SecureAddHost(data_y)

    def run(self):
        self.secure_add_host_inst.run()

    
if __name__ == "__main__":
    jobid = sys.argv[1]
    guest_partyid = sys.argv[2]
    host_partyid = sys.argv[3]

    work_mode = 0
    if len(sys.argv) > 4:
        work_mode = int(sys.argv[4])

    secure_add_host_workflow = SecureAddHostWorkflow(jobid,
                                                     guest_partyid,
                                                     host_partyid,
                                                     work_mode)
    
    secure_add_host_workflow.run()
