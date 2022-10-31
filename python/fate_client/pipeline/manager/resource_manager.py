from ..utils.id_gen import get_uuid
from ..conf.env_config import FateStandaloneConfig
from . import get_data_manager, get_model_manager, get_metric_manager, get_status_manager, get_job_conf_manager


class ResourceManager(object):
    def __init__(self, *args, **kwargs):
        ...


class FateStandaloneResourceManager(object):
    def __init__(self, conf: 'FateStandaloneConfig'):
        self._conf = conf
        self._data_namespace = "output_data"
        self._data_name = get_uuid()
        self._data_manager = get_data_manager(conf.OUTPUT_DATA_DIR)
        self._model_manager = get_model_manager(conf.OUTPUT_MODEL_DIR)
        self._metric_manager = get_metric_manager(conf.OUTPUT_METRIC_DIR)
        self._status_manager = get_status_manager(conf.OUTPUT_STATUS_DIR)
        self._job_conf_manager = get_job_conf_manager(conf.JOB_DIR)

    def generate_task_runtime_conf(self, session_id: str, computing_id: str, federation_id: str, job_type: str,
                                   module: str, runtime_parties: dict, role: str,
                                   party: str, params: dict, inputs: dict, outputs: dict, backends: dict):

        task_runtime_conf = self._job_conf_manager.construct_job_runtime_conf(
            task_id=session_id,
            job_type=job_type,
            module=module,
            runtime_parties=runtime_parties,
            role=role,
            party_id=party,
            params=params,
            inputs=inputs,
            outputs=outputs,
            backends=backends,
            computing_id=computing_id,
            federation_id=federation_id
        )

        return task_runtime_conf

    def generate_task_outputs(self, session_id: str, role: str, party: str, outputs: dict):
        output_resource = dict()
        if outputs:
            for output_outer_key, output_inner_keys in outputs.items():
                output_resource[output_outer_key] = dict()
                if output_outer_key == "data":
                    for data_key in output_inner_keys:
                        output_data_uri = self._generate_output_data_uri(session_id, role, party, data_key)
                        output_resource[output_outer_key][data_key] = output_data_uri
                elif output_outer_key == "model":
                    for model_key in output_inner_keys:
                        output_model_uri = self._generate_output_model_uri(session_id, role, party, model_key)
                        output_resource[output_outer_key][model_key] = output_model_uri
                else:
                    raise ValueError(f"Unsupported output key: {output_outer_key}")

            output_resource["metric"] = self._generate_output_metric_uri(session_id, role, party)
        output_resource["status"] = self._generate_output_status_uri(session_id, role, party)

        return output_resource

    def _generate_output_data_uri(self, session_id, role, party, data_suffix):
        return self._data_manager.generate_output_data_uri(self._conf.OUTPUT_DATA_DIR,
                                                           session_id,
                                                           role,
                                                           party,
                                                           data_suffix,
                                                           self._data_namespace,
                                                           self._data_name)

    def _generate_output_model_uri(self, session_id, role, party, model_key):
        return self._model_manager.generate_output_model_uri(self._conf.OUTPUT_MODEL_DIR,
                                                             session_id,
                                                             role,
                                                             party,
                                                             model_key)

    def _generate_output_metric_uri(self, session_id, role, party):
        return self._metric_manager.generate_output_metric_uri(self._conf.OUTPUT_METRIC_DIR,
                                                               session_id,
                                                               role,
                                                               party)

    def _generate_output_status_uri(self, session_id, role, party):
        return self._status_manager.generate_output_status_uri(self._conf.OUTPUT_STATUS_DIR,
                                                               session_id,
                                                               role,
                                                               party)

    def generate_job_conf_uri(self, session_id, role, party):
        return self._job_conf_manager.generate_job_conf_uri(self._conf.JOB_DIR,
                                                            session_id,
                                                            role,
                                                            party)

    def write_out_job_conf(self, job_conf_uri: str, job_conf: dict):
        self._job_conf_manager.record_job_conf(job_conf_uri, job_conf)

    @property
    def status_manager(self):
        return self._status_manager
