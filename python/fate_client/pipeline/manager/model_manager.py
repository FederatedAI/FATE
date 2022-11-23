from ..utils.uri_tools import parse_uri, replace_uri_path, get_schema_from_uri
from ..utils.file_utils import construct_local_dir
from ..conf.types import UriTypes


class LocalFSModelManager(object):
    @classmethod
    def generate_output_model_uri(cls, output_dir_uri: str, job_id: str, task_name: str,
                                  role: str, party_id: str, model_suffix: str):
        model_id = "_".join([job_id, task_name, role, str(party_id), model_suffix])
        model_version = "v0"
        suffix = "model.json"
        uri_obj = parse_uri(output_dir_uri)
        local_path = construct_local_dir(uri_obj.path, *[model_id, model_version, suffix])
        uri_obj = replace_uri_path(uri_obj, str(local_path))
        return uri_obj.geturl()


class LMDBModelManager(object):
    @classmethod
    def generate_output_model_uri(cls, uri_obj, session_id: str, role: str, party_id: str, namespace: str, name: str):
        ...


def get_model_manager(model_uri: str):
    uri_type = get_schema_from_uri(model_uri)
    if uri_type == UriTypes.LOCAL:
        return LocalFSModelManager
    else:
        return LMDBModelManager

