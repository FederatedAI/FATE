import json
import os
from pathlib import Path
from ..utils.uri_tools import parse_uri, replace_uri_path, get_schema_from_uri
from ..utils.file_utils import construct_local_dir, write_json_file
from ..conf.types import UriTypes


class LocalFSStatusManager(object):
    @classmethod
    def generate_output_status_uri(cls, output_dir_uri: str, session_id: str,
                                   role: str, party_id: str):
        uri_obj = parse_uri(output_dir_uri)
        local_path = construct_local_dir(uri_obj.path, *[session_id, role, party_id, "status.log"])
        uri_obj = replace_uri_path(uri_obj, str(local_path))
        return uri_obj.geturl()

    @classmethod
    def monitor_status(cls, status_uris):
        for status_uri in status_uris:
            uri_obj = parse_uri(status_uri.status_uri)
            if not os.path.exists(uri_obj.path):
                return False

        return True

    @classmethod
    def record_finish_status(cls, status_uri):
        uri_obj = parse_uri(status_uri)
        path = Path(uri_obj.path).parent.joinpath("done")
        buf = dict(job_status="done")

        write_json_file(str(path), buf)

    @classmethod
    def get_tasks_status(cls, task_status_uris):
        summary_msg = dict()
        summary_status = "SUCCESS"
        for obj in task_status_uris:
            try:
                path = parse_uri(obj.status_uri).path
                with open(path, "r") as fin:
                    party_status = json.loads(fin.read())

                if party_status["status"]["status"] != "SUCCESS":
                    summary_status = "FAIL"
            except FileNotFoundError:
                party_status = dict(
                    status=dict(
                        status="FAIL",
                        extras="can not start task"
                    )
                )

            if obj.role not in summary_msg:
                summary_msg[obj.role] = dict()
            summary_msg[obj.role][obj.party_id] = party_status

        ret = dict(summary_status=summary_status,
                   retmsg=summary_msg)

        return ret


class LMDBStatusManager(object):
    @classmethod
    def generate_output_status_uri(cls, uri_obj, session_id: str, role: str, party_id: str):
        ...

    @classmethod
    def record_finish_status(cls, status_uri):
        ...

    @classmethod
    def get_task_status(cls, status_uris):
        ...

    @classmethod
    def monitor_status(cls, ):
        ...


def get_status_manager(model_uri: str):
    uri_type = get_schema_from_uri(model_uri)
    if uri_type == UriTypes.LOCAL:
        return LocalFSStatusManager
    else:
        return LMDBStatusManager
