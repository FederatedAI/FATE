import json
from typing import Union
from ..utils.uri_tools import parse_uri, replace_uri_path, get_schema_from_uri
from ..utils.file_utils import construct_local_dir
from ..conf.types import UriTypes


class LocalFSMetricManager(object):
    @classmethod
    def generate_output_metric_uri(cls, output_dir_uri: str, job_id: str, task_name: str,
                                   role: str, party_id: Union[str, int]):
        uri_obj = parse_uri(output_dir_uri)
        namespace = "_".join([job_id, task_name, role, str(party_id)])
        local_path = construct_local_dir(uri_obj.path, *[namespace])
        uri_obj = replace_uri_path(uri_obj, str(local_path))
        return uri_obj.geturl()

    @classmethod
    def get_output_metrics(cls, uri):
        uri_obj = parse_uri(uri)
        with open(uri_obj.path, "r") as fin:
            return json.loads(fin.read())


def get_metric_manager(metric_uri: str):
    uri_type = get_schema_from_uri(metric_uri)
    if uri_type == UriTypes.LOCAL:
        return LocalFSMetricManager
