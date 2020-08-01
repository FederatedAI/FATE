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
from fate_flow.settings import API_VERSION
from fate_flow.utils import api_utils
from fate_flow.api.client.controller import api_client


class ControllerRemoteClient(api_client.ControllerClient):
    @classmethod
    def update_job(cls, job_info):
        response = api_utils.local_api(
            job_id=job_info["job_id"],
            method='POST',
            endpoint='/{}/control/{}/{}/{}/update'.format(
                API_VERSION,
                job_info["job_id"],
                job_info["role"],
                job_info["party_id"]
            ),
            json_body=job_info)
        return response

    @classmethod
    def update_task_set(cls, task_set_info):
        response = api_utils.local_api(
            job_id=task_set_info["job_id"],
            method='POST',
            endpoint='/{}/control/{}/{}/{}/{}/update'.format(
                API_VERSION,
                task_set_info["job_id"],
                task_set_info["task_set_id"],
                task_set_info["role"],
                task_set_info["party_id"]
            ),
            json_body=task_set_info)
        return response

    @classmethod
    def update_task(cls, task_info):
        response = api_utils.local_api(
            job_id=task_info["job_id"],
            method='POST',
            endpoint='/{}/control/{}/{}/{}/{}/{}/{}/update'.format(
                API_VERSION,
                task_info["job_id"],
                task_info["component_name"],
                task_info["task_id"],
                task_info["task_version"],
                task_info["role"],
                task_info["party_id"]
            ),
            json_body=task_info)
        return response
