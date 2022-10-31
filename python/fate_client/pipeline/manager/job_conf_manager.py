from typing import Union
from ..utils.uri_tools import parse_uri, replace_uri_path, get_schema_from_uri
from ..utils.file_utils import construct_local_dir, write_json_file
from ..conf.types import UriTypes


class JobConfManager(object):
    @classmethod
    def construct_job_runtime_conf(cls, task_id: str, job_type: str, module: str, runtime_parties: dict,
                                   role: str, party_id: str, params: dict, inputs: Union[dict],
                                   outputs: Union[dict], backends: Union[dict, None], computing_id=None,
                                   federation_id=None) -> dict:
        job_runtime_conf = dict()
        job_runtime_conf["task_id"] = task_id
        job_runtime_conf["task"] = {"type": job_type}
        task_params = dict(
            cpn=module,
            role=role,
            party_id=party_id,
            params=params
        )

        # TODO: input and output should be optimize, now just for mini-demo, dict(output_type=uri) maybe better
        data_keys = ["data", "train_data", "validate_data", "test_data"]
        task_params["data_inputs"] = []
        for data_key in data_keys:
            if data_key in inputs:
                task_params["data_inputs"] = [list(inputs[data_key].items())[0][1]]

        if "model" in inputs:
            task_params["model_inputs"] = inputs["model"]
        else:
            task_params["model_inputs"] = []

        if "data" in outputs:
            # task_params["data_outputs"] = outputs["data"]
            task_params["data_outputs"] = [list(outputs["data"].items())[0][1]]
        else:
            task_params["data_outputs"] = []

        if "model" in outputs:
            # task_params["model_outputs"] = outputs["model"]
            task_params["model_outputs"] = [list(outputs["model"].items())[0][1]]
        else:
            task_params["model_outputs"] = []

        task_params["status_output"] = outputs["status"]
        task_params["metrics_output"] = outputs["metric"]

        job_runtime_conf["task"]["task_params"] = task_params
        job_runtime_conf["task"]["task_extra"] = cls.construct_extra_runtime_conf(
            role,
            party_id,
            runtime_parties,
            backends,
            computing_id,
            federation_id
        )

        return job_runtime_conf

    @classmethod
    def construct_extra_runtime_conf(cls, role: str, party_id: str, runtime_parties: dict, backends: Union[dict, None],
                                     computing_id: Union[None, str], federation_id: Union[None, str]):
        task_extra = dict()
        if backends:
            task_extra["device"] = backends["device"]
            task_extra["distributed_computing_backend"] = dict(
                engine=backends["computing_engine"],
                computing_id=computing_id
            )

            # TODO: this may need to optimize, now just for mini-demo
            parties = []
            for _role, party_id_list in runtime_parties.items():
                for _party_id in party_id_list:
                    parties.append([_role, _party_id])

            task_extra["federation_backend"] = dict(
                engine=backends["federation_engine"],
                federation_id=federation_id,
                parties=dict(
                    parties=parties,
                    local=[role, party_id]
                )

            )

        return task_extra


class LocalFSJobConfManager(JobConfManager):
    FILE_SUFFIX = "job_runtime_conf.json"

    @classmethod
    def generate_job_conf_uri(cls, output_dir_uri: str, session_id: str,
                              role: str, party_id: str):
        uri_obj = parse_uri(output_dir_uri)
        local_path = construct_local_dir(uri_obj.path, *[session_id, role, party_id, cls.FILE_SUFFIX])
        uri_obj = replace_uri_path(uri_obj, str(local_path))
        return uri_obj.geturl()

    @classmethod
    def record_job_conf(cls, uri, job_conf):
        write_json_file(uri, job_conf)


class LMDBJobConfManager(JobConfManager):
    @classmethod
    def generate_job_conf_uri(cls, uri_obj, session_id: str, role: str, party_id: str):
        ...


def get_job_conf_manager(job_conf_uri: str):
    uri_type = get_schema_from_uri(job_conf_uri)
    if uri_type == UriTypes.LOCAL:
        return LocalFSJobConfManager
    else:
        return LMDBJobConfManager
