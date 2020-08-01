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
class JobTrackerClient(object):
    def __init__(self, job_id: str, role: str, party_id: int,
                 model_id: str = None,
                 model_version: str = None,
                 task_set_id: int = None,
                 component_name: str = None,
                 component_module_name: str = None,
                 task_id: str = None,
                 task_version: int = None
                 ):
        self.job_id = job_id
        self.role = role
        self.party_id = party_id
        self.model_id = model_id
        self.party_model_id = model_utils.gen_party_model_id(model_id=model_id, role=role, party_id=party_id)
        self.model_version = model_version
        self.pipelined_model = None
        if self.party_model_id and self.model_version:
            self.pipelined_model = pipelined_model.PipelinedModel(model_id=self.party_model_id,
                                                                  model_version=self.model_version)

        self.task_set_id = task_set_id

        self.component_name = component_name if component_name else 'pipeline'
        self.module_name = component_module_name if component_module_name else 'Pipeline'
        self.task_id = task_id
        self.task_version = task_version
        self.table_namespace = '_'.join(
            ['fate_flow', 'tracking', 'data', self.job_id, self.role, str(self.party_id), self.component_name])
        self.job_table_namespace = '_'.join()

    def log_job_metric_data(self, metric_namespace: str, metric_name: str, metrics: List[Metric]):
        pass

    def log_metric_data(self, metric_namespace: str, metric_name: str, metrics: List[Metric]):
        pass

