from ..utils.uri_tools import parse_uri, replace_uri_path, get_schema_from_uri
from ..utils.file_utils import construct_local_dir
from ..conf.types import UriTypes


class LocalFSDataManager(object):
    @classmethod
    def generate_output_data_uri(cls, output_dir_uri: str, session_id: str,
                                 role: str, party_id: str, data_suffix: str, namespace: str, name: str):
        uri_obj = parse_uri(output_dir_uri)
        local_path = construct_local_dir(uri_obj.path, *[session_id, role, party_id, data_suffix, namespace, name])
        uri_obj = replace_uri_path(uri_obj, str(local_path))
        return uri_obj.geturl()


class LMDBDataManager(object):
    @classmethod
    def generate_output_data_uri(cls, output_dir_uri: str, session_id: str,
                                 role: str, party_id: str, data_suffix: str, namespace: str, name: str):
        ...


def get_data_manager(output_dir_uri):
    uri_type = get_schema_from_uri(output_dir_uri)
    if uri_type == UriTypes.LOCAL:
        return LocalFSDataManager
    elif uri_type == UriTypes.LMDB:
        return LMDBDataManager
