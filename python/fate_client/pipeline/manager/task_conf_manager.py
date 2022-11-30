from ..utils.uri_tools import parse_uri, get_schema_from_uri
from ..utils.file_utils import construct_local_dir, write_yaml_file
from ..conf.types import UriTypes


class LocalFSTaskConfManager(object):
    FILE_SUFFIX = "task_runtime_conf.yaml"

    @classmethod
    def generate_task_conf_uri(cls, output_dir_uri: str, job_id: str, task_name: str,
                               role: str, party_id: str):
        uri_obj = parse_uri(output_dir_uri)
        local_path = construct_local_dir(uri_obj.path, *[job_id, task_name, role, party_id, cls.FILE_SUFFIX])
        return str(local_path)

    @classmethod
    def record_task_conf(cls, output_dir_uri, job_id, task_name, role, party_id, task_conf):
        path = cls.generate_task_conf_uri(output_dir_uri, job_id, task_name, role, party_id)
        write_yaml_file(path, task_conf)

        return path


class LMDBTaskConfManager(object):
    ...


def get_task_conf_manager(job_conf_uri: str):
    uri_type = get_schema_from_uri(job_conf_uri)
    if uri_type == UriTypes.LOCAL:
        return LocalFSTaskConfManager
    else:
        return LMDBTaskConfManager
