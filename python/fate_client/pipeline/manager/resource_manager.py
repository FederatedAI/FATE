from ..utils.id_gen import get_uuid
from ..utils.file_utils import construct_local_dir
from ..conf.env_config import StandaloneConfig
from ..entity.task_structure import IOArtifact
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
        self._task_conf_manager = get_task_conf_manager(conf.JOB_DIR)

    def generate_output_artifact(self, job_id, task_name, role, party_id, output_key, artifact_type):
        if artifact_type in ["model", "models"]:
            model_uri = self._generate_output_model_uri(
                job_id,
                task_name,
                role,
                party_id,
                output_key
            )

            return IOArtifact(
                name=output_key,
                uri=model_uri,
                metadata=dict(format="json")
            )
        elif artifact_type in ["dataset", "datasets"]:
            data_uri = self._generate_output_data_uri(
                job_id,
                task_name,
                role,
                party_id,
                output_key
            )

            return IOArtifact(
                name=output_key,
                uri=data_uri,
                metadata=dict(format="dataframe")
            )
        else:
            uri = self._generate_output_metric_uri(
                job_id,
                task_name,
                role,
                party_id,
                output_key
            )
            return IOArtifact(
                name="ClassificationMetrics",
                uri=uri,
                metadata=dict(format="json")
            )

    def _generate_output_data_uri(self, job_id, task_name, role, party_id, output_key):
        """
        $job_id_${task_name}_$role_${party_id}/output_key
        """
        return self._data_manager.generate_output_data_uri(self._conf.OUTPUT_DATA_DIR,
                                                           job_id,
                                                           task_name,
                                                           role,
                                                           party_id,
                                                           output_key)

    def _generate_output_model_uri(self, job_id, task_name, role, party_id, output_key):
        """
        model_id/model_version
        model_id=${job_id}_${task_name}_$role_${party_id}_${output_key}/v0
        """
        return self._model_manager.generate_output_model_uri(self._conf.OUTPUT_MODEL_DIR,
                                                             job_id,
                                                             task_name,
                                                             role,
                                                             party_id,
                                                             output_key)

    def _generate_output_metric_uri(self, job_id, task_name, role, party_id, output_key):
        return self._metric_manager.generate_output_metric_uri(self._conf.OUTPUT_METRIC_DIR,
                                                               job_id,
                                                               task_name,
                                                               role,
                                                               party_id,
                                                               output_key)

    def generate_output_terminate_status_uri(self, job_id, task_name, role, party_id):
        return self._status_manager.generate_output_terminate_status_uri(self._conf.OUTPUT_STATUS_DIR,
                                                                         job_id,
                                                                         task_name,
                                                                         role,
                                                                         party_id)

    @staticmethod
    def generate_log_uri(log_dir_prefix, role, party_id):
        return str(construct_local_dir(log_dir_prefix, *[role, str(party_id)]))

    def write_out_task_conf(self, job_id, task_name, role, party_id, task_conf):
        task_conf_uri = self._task_conf_manager.record_task_conf(
            self._conf.JOB_DIR,
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
