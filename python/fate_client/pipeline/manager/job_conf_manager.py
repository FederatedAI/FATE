from ..utils.uri_tools import parse_uri, replace_uri_path, get_schema_from_uri
from ..utils.file_utils import construct_local_dir, write_json_file
from ..conf.types import UriTypes


class LocalFSJobConfManager(object):
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


class LMDBJobConfManager(object):
    @classmethod
    def generate_job_conf_uri(cls, uri_obj, session_id: str, role: str, party_id: str):
        ...


def get_job_conf_manager(job_conf_uri: str):
    uri_type = get_schema_from_uri(job_conf_uri)
    if uri_type == UriTypes.LOCAL:
        return LocalFSJobConfManager
    else:
        return LMDBJobConfManager
