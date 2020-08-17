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


class TaskInfo(object):
    def __init__(self, jobid, component, job_client, role='guest', party_id=9999):
        self._jobid = jobid
        self._component = component
        self._job_client = job_client
        self._party_id = party_id
        self._role = role

    def get_output_data(self, limits=None):
        return self._job_client.get_output_data(self._jobid, self._component.name, self._role, self._party_id, limits)

    def get_model_param(self):
        return self._job_client.get_model_param(self._jobid, self._component.name, self._role, self._party_id)

    def get_output_data_table(self):
        return self._job_client.get_output_data_table(self._jobid, self._component.name, self._role, self._party_id)

    def get_summary(self):
        return self._job_client.get_summary(self._jobid, self._component.name, self._role, self._party_id)

