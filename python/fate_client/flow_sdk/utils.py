import os
import json
import socket
import time
import typing
import tarfile
import datetime
from enum import Enum, IntEnum

PROJECT_BASE = os.getenv("FATE_DEPLOY_BASE")


def start_cluster_standalone_job_server():
    print("use service.sh to start standalone node server....")
    os.system("sh service.sh start --standalone_node")
    time.sleep(5)


def get_parser_version_set():
    return {"1", "2"}


def get_project_base_directory():
    global PROJECT_BASE
    if PROJECT_BASE is None:
        PROJECT_BASE = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir
            )
        )
    return PROJECT_BASE


def download_from_request(http_response, tar_file_name, extract_dir):
    with open(tar_file_name, "wb") as fw:
        for chunk in http_response.iter_content(1024):
            if chunk:
                fw.write(chunk)
    tar = tarfile.open(tar_file_name, "r:gz")
    file_names = tar.getnames()
    for file_name in file_names:
        tar.extract(file_name, extract_dir)
    tar.close()
    os.remove(tar_file_name)


def check_config(config: typing.Dict, required_arguments: typing.List):
    no_arguments = []
    error_arguments = []
    for require_argument in required_arguments:
        if isinstance(require_argument, tuple):
            config_value = config.get(require_argument[0], None)
            if isinstance(require_argument[1], (tuple, list)):
                if config_value not in require_argument[1]:
                    error_arguments.append(require_argument)
            elif config_value != require_argument[1]:
                error_arguments.append(require_argument)
        elif require_argument not in config:
            no_arguments.append(require_argument)
    if no_arguments or error_arguments:
        raise Exception(
            "the following arguments are required: {} {}".format(
                ",".join(no_arguments),
                ",".join(["{}={}".format(a[0], a[1]) for a in error_arguments]),
            )
        )


def preprocess(**kwargs):
    kwargs.pop('self', None)
    kwargs.pop('kwargs', None)
    config_data = kwargs.pop('config_data', {})
    dsl_data = kwargs.pop('dsl_data', {})

    output_path = kwargs.pop('output_path', None)
    if output_path is not None:
        config_data['output_path'] = os.path.abspath(output_path)

    local = config_data.pop('local', {})
    party_id = kwargs.pop('party_id', None)
    role = kwargs.pop('role', None)
    if party_id is not None:
        kwargs['party_id'] = local['party_id'] = int(party_id)
    if role is not None:
        kwargs['role'] = local['role'] = role
    if local:
        config_data['local'] = local

    for k, v in kwargs.items():
        if v is not None:
            if k in {'job_id', 'model_version'}:
                v = str(v)
            elif k in {'party_id', 'step_index'}:
                v = int(v)
            config_data[k] = v

    return config_data, dsl_data


def check_output_path(path):
    if not os.path.isabs(path):
        return os.path.join(os.path.abspath(os.curdir), path)
    return path


def string_to_bytes(string):
    return string if isinstance(string, bytes) else string.encode(encoding="utf-8")


def get_lan_ip():
    if os.name != "nt":
        import fcntl
        import struct

        def get_interface_ip(ifname):
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            return socket.inet_ntoa(
                fcntl.ioctl(
                    s.fileno(),
                    0x8915,
                    struct.pack("256s", string_to_bytes(ifname[:15])),
                )[20:24]
            )

    ip = socket.gethostbyname(socket.getfqdn())
    if ip.startswith("127.") and os.name != "nt":
        interfaces = [
            "bond1",
            "eth0",
            "eth1",
            "eth2",
            "wlan0",
            "wlan1",
            "wifi0",
            "ath0",
            "ath1",
            "ppp0",
        ]
        for ifname in interfaces:
            try:
                ip = get_interface_ip(ifname)
                break
            except IOError as e:
                pass
    return ip or ""


class CustomJSONEncoder(json.JSONEncoder):
    def __init__(self, **kwargs):
        super(CustomJSONEncoder, self).__init__(**kwargs)

    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(obj, datetime.date):
            return obj.strftime("%Y-%m-%d")
        elif isinstance(obj, datetime.timedelta):
            return str(obj)
        elif issubclass(type(obj), Enum) or issubclass(type(obj), IntEnum):
            return obj.value
        else:
            return json.JSONEncoder.default(self, obj)


def json_dumps(src, byte=False, indent=None):
    if byte:
        return string_to_bytes(json.dumps(src, indent=indent, cls=CustomJSONEncoder))
    else:
        return json.dumps(src, indent=indent, cls=CustomJSONEncoder)
