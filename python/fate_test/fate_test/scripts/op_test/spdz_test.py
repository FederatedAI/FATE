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
import json
import time
from prettytable import PrettyTable, ORGMODE
from flow_sdk.client import FlowClient


class SPDZTest(object):
    def __init__(self, flow_address, params, conf_path, dsl_path, guest_party_id, host_party_id):
        self.client = FlowClient(ip=flow_address.split(":")[0],
                                 port=flow_address.split(":")[1],
                                 version="v1")

        self.dsl = self._get_json_file(dsl_path)
        self.conf = self._get_json_file(conf_path)
        self.conf["role"] = dict(guest=guest_party_id, host=host_party_id)
        self.conf["component_parameters"]["common"]["spdz_test_0"].update(params)
        self.conf["initiator"]["party_id"] = guest_party_id[0]
        self.guest_party_id = guest_party_id[0]

    @staticmethod
    def _get_json_file(path):
        with open(path, "r") as fin:
            ret = json.loads(fin.read())

        return ret

    def run(self):
        result = self.client.job.submit(config_data=self.conf, dsl_data=self.dsl)

        try:
            if 'retcode' not in result or result["retcode"] != 0:
                raise ValueError(f"retcode err")

            if "jobId" not in result:
                raise ValueError(f"jobID not in result: {result}")

            job_id = result["jobId"]
        except ValueError:
            raise ValueError("job submit failed, err msg: {}".format(result))

        while True:
            info = self.client.job.query(job_id=job_id, role="guest", party_id=self.guest_party_id)
            data = info["data"][0]
            status = data["f_status"]
            if status == "success":
                break
            elif status == "failed":
                raise ValueError(f"job is failed, jobid is {job_id}")

            time.sleep(1)

        summary = self.client.component.get_summary(job_id=job_id, role="guest",
                                                    party_id=self.guest_party_id,
                                                    component_name="spdz_test_0")

        summary = summary["data"]
        field_name = summary["field_name"]

        tables = []
        for tensor_type in summary["tensor_type"]:
            table = PrettyTable()
            table.set_style(ORGMODE)
            table.field_names = field_name
            for op_type in summary["op_test_list"]:
                table.add_row(summary[tensor_type][op_type])

            tables.append(table.get_string(title=f"SPDZ {tensor_type} Computational performance"))

        return tables
