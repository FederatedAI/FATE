from typing import Union
from ..utils.uri_tools import parse_uri, replace_uri_path, get_schema_from_uri
from ..utils.file_utils import construct_local_dir
from ..conf.types import UriTypes


class LocalFSDataManager(object):
    @classmethod
    def generate_output_data_uri(cls, output_dir_uri: str, job_id: str, task_name: str,
                                 role: str, party_id: Union[str, int], data_suffix: str):
        uri_obj = parse_uri(output_dir_uri)
        namespace = "_".join([job_id, task_name, role, str(party_id)])
        name = data_suffix
        local_path = construct_local_dir(uri_obj.path, *[namespace, name])
        uri_obj = replace_uri_path(uri_obj, str(local_path))
        return uri_obj.geturl()


class LMDBDataManager(object):
    @classmethod
    def generate_output_data_uri(cls, output_dir_uri: str, job_id: str, task_name: str,
                                 role: str, party_id: Union[str, int], data_suffix: str):
        ...


def get_data_manager(output_dir_uri):
    uri_type = get_schema_from_uri(output_dir_uri)
    if uri_type == UriTypes.LOCAL:
        return LocalFSDataManager
    elif uri_type == UriTypes.LMDB:
        return LMDBDataManager
