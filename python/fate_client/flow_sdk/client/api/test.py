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
import os.path

from flow_sdk.client.api.base import BaseFlowAPI
from flow_sdk.utils import preprocess


class Test(BaseFlowAPI):
    def toy(self, guest_party_id: str, host_party_id: str, guest_user_name: str = "", host_user_name: str = "",
            task_cores: int = 2, timeout: int = 60):
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        conf = self.toy_conf(**kwargs)
        return self._post(url='job/submit', json={
            'job_runtime_conf': conf,
            'job_dsl': self.toy_dsl(),
        })

    @classmethod
    def toy_conf(cls, guest_party_id: str, host_party_id: str, guest_user_name: str = "", host_user_name: str = "",
                 task_cores: int = 2, **kwargs):
        job_conf = {
            "dsl_version": 2,
            "job_parameters": {
            },
            "role": {
                "guest": [
                    guest_party_id
                ],
                "host": [
                    host_party_id
                ]
            },
            "component_parameters": {
                "role": {
                    "guest": {
                        "0": {
                            "secure_add_example_0": {
                                "seed": 123
                            }
                        }
                    },
                    "host": {
                        "secure_add_example_0": {
                            "seed": 321
                        }
                    }
                },
                "common": {
                    "secure_add_example_0": {
                        "partition": 4,
                        "data_num": 1000
                    }
                }
            }
        }
        job_conf["initiator"] = {
            "role": "guest",
            "party_id": guest_party_id
        }
        job_conf["role"]["guest"] = [guest_party_id]
        job_conf["role"]["host"] = [host_party_id]
        job_conf["job_parameters"]["common"] = {
            "task_cores": task_cores
        }
        job_conf["job_parameters"]["role"] = {
            "guest": {"0": {"user": guest_user_name}},
            "host": {"0": {"user": host_user_name}}
        }
        return job_conf

    @classmethod
    def toy_dsl(cls):
        dsl = {
            "components": {
                "secure_add_example_0": {
                    "module": "SecureAddExample"
                }
            }
        }
        return dsl

    @classmethod
    def check_toy(cls, guest_party_id, job_status, log_dir):
        if job_status in {"success", "canceled"}:
            info_log = os.path.join(log_dir, "guest", guest_party_id, "INFO.log")
            with open(info_log, "r") as fin:
                for line in fin:
                    if line.find("secure_add_guest") != -1:
                        yield line.strip()
        else:
            error_log = os.path.join(log_dir, "guest", guest_party_id, "ERROR.log")
            with open(error_log, "r") as fin:
                for line in fin:
                    yield line.strip()
