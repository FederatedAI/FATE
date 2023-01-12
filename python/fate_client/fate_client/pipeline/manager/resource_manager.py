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
from ..utils.standalone.id_gen import get_uuid
from ..utils.file_utils import construct_local_dir
from ..conf.env_config import StandaloneConfig
from ..entity.task_structure import OutputArtifact
from . import get_data_manager, get_model_manager, get_metric_manager, get_status_manager, get_task_conf_manager


class StandaloneResourceManager(object):
    def __init__(self, conf: 'StandaloneConfig'):
        self._conf = conf
        self._data_namespace = "output_data"
        self._data_name = get_uuid()
        self._data_manager = get_data_manager(conf.OUTPUT_DATA_DIR)
        self._model_manager = get_model_manager(conf.OUTPUT_MODEL_DIR)
        self._metric_manager = get_metric_manager(conf.OUTPUT_METRIC_DIR)
        self._status_manager = get_status_manager().create_status_manager(conf.MLMD.db)
        self._task_conf_manager = get_task_conf_manager(conf.JOB_CONF_DIR)

    def generate_output_artifact(self, job_id, task_name, role, party_id, artifact_type):
        if artifact_type == "model":
            model_uri = self._generate_output_model_uri(
                job_id,
                task_name,
                role,
                party_id
            )

            return OutputArtifact(
                type="directory",
                metadata=dict(uri=model_uri,
                              format="json"
                              )
            )
        elif artifact_type == "data":
            data_uri = self._generate_output_data_uri(
                job_id,
                task_name,
                role,
                party_id
            )

            return OutputArtifact(
                type="directory",
                metadata=dict(uri=data_uri,
                              format="dataframe"
                              )
            )
        else:
            uri = self._generate_output_metric_uri(
                job_id,
                task_name,
                role,
                party_id
            )
            return OutputArtifact(
                type="directory",
                metadata=dict(uri=uri,
                              format="json")
            )

    def _generate_output_data_uri(self, job_id, task_name, role, party_id):
        """
        $job_id_${task_name}_$role_${party_id}/output_key
        """
        return self._data_manager.generate_output_data_uri(self._conf.OUTPUT_DATA_DIR,
                                                           job_id,
                                                           task_name,
                                                           role,
                                                           party_id)

    def _generate_output_model_uri(self, job_id, task_name, role, party_id):
        """
        model_id/model_version
        model_id=${job_id}_${task_name}_$role_${party_id}_${output_key}/v0
        """
        return self._model_manager.generate_output_model_uri(self._conf.OUTPUT_MODEL_DIR,
                                                             job_id,
                                                             task_name,
                                                             role,
                                                             party_id)

    def _generate_output_metric_uri(self, job_id, task_name, role, party_id):
        return self._metric_manager.generate_output_metric_uri(self._conf.OUTPUT_METRIC_DIR,
                                                               job_id,
                                                               task_name,
                                                               role,
                                                               party_id)

    def generate_output_terminate_status_uri(self, job_id, task_name, role, party_id):
        return self._status_manager.generate_output_terminate_status_uri(self._conf.OUTPUT_STATUS_DIR,
                                                                         job_id,
                                                                         task_name,
                                                                         role,
                                                                         party_id)

    def get_output_data(self, uri):
        return self._data_manager.get_output_data(uri)

    def get_output_model(self, uri):
        return self._model_manager.get_output_model(uri)

    def get_output_metrics(self, uri):
        return self._metric_manager.get_output_metrics(uri)

    @staticmethod
    def generate_log_uri(log_dir_prefix, role, party_id):
        return str(construct_local_dir(log_dir_prefix, *[role, str(party_id)]))

    def write_out_task_conf(self, job_id, task_name, role, party_id, task_conf):
        task_conf_uri = self._task_conf_manager.record_task_conf(
            self._conf.JOB_CONF_DIR,
            job_id,
            task_name,
            role,
            party_id,
            task_conf
        )

        return task_conf_uri

    @property
    def status_manager(self):
        return self._status_manager
