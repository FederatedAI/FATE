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
    config_data = {}
    if "self" in kwargs:
        kwargs.pop("self")

    if kwargs.get("conf_path"):
        conf_path = os.path.abspath(kwargs.get("conf_path"))
        with open(conf_path, "r") as conf_fp:
            config_data = json.load(conf_fp)

        if config_data.get("output_path"):
            config_data["output_path"] = os.path.abspath(config_data["output_path"])

        if ("party_id" in kwargs.keys()) or ("role" in kwargs.keys()):
            config_data["local"] = config_data.get("local", {})
            if kwargs.get("party_id"):
                config_data["local"]["party_id"] = kwargs.get("party_id")
            if kwargs.get("role"):
                config_data["local"]["role"] = kwargs.get("role")

    config_data.update(dict((k, v) for k, v in kwargs.items() if v is not None))

    for key in ["job_id", "party_id"]:
        if isinstance(config_data.get(key), int):
            config_data[key] = str(config_data[key])

    dsl_data = {}
    if kwargs.get("dsl_path"):
        dsl_path = os.path.abspath(kwargs.get("dsl_path"))
        with open(dsl_path, "r") as dsl_fp:
            dsl_data = json.load(dsl_fp)
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
