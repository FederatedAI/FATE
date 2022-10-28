from ..utils.uri_tools import parse_uri, replace_uri_path, get_schema_from_uri
from ..utils.file_utils import construct_local_dir
from ..conf.types import UriTypes


class LocalFSMetricManager(object):
    @classmethod
    def generate_output_metric_uri(cls, output_dir_uri: str, session_id: str,
                                   role: str, party_id: str):
        uri_obj = parse_uri(output_dir_uri)
        local_path = construct_local_dir(uri_obj.path, *[session_id, role, party_id])
        uri_obj = replace_uri_path(uri_obj, str(local_path))
        return uri_obj.geturl()


class LMDBDataManager(object):
    @classmethod
    def generate_output_metric_uri(cls, output_dir_uri, session_id: str, role: str, party_id: str):
        ...


def get_metric_manager(metric_uri: str):
    uri_type = get_schema_from_uri(metric_uri)
    if uri_type == UriTypes.LOCAL:
        return LocalFSMetricManager
    else:
        return LMDBDataManager
