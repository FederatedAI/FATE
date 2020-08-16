#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import json
import os
from pathlib import Path

from ruamel import yaml

from flow_sdk.client.base import BaseFlowClient
from flow_sdk.client import api

from flow_sdk.utils import get_lan_ip


def parse_config():
    with Path(__file__).parent.parent.parent.joinpath("flow_client").joinpath("settings.yaml").open() as fin:
        config = yaml.safe_load(fin)
    if config.get("server_conf_path"):
        is_server_conf_exist = os.path.exists(config.get("server_conf_path"))
    else:
        is_server_conf_exist = False

    if is_server_conf_exist:
        try:
            with open(config.get("server_conf_path")) as server_conf_fp:
                server_conf = json.load(server_conf_fp)
            ip = server_conf.get(config.get("server")).get(config.get("role")).get("host")
            if ip in ["localhost", "127.0.0.1"]:
                ip = get_lan_ip()
            http_port = server_conf.get(config.get("server", None)).get(config.get("role", None)).get("http.port", None)
            api_version = config.get("api_version")
        except Exception:
            return
    elif config.get("ip") and config.get("port"):
        ip = config.get("ip")
        if ip in ["localhost", "127.0.0.1"]:
            ip = get_lan_ip()
        http_port = int(config.get("port"))
        api_version = config.get("api_version")
    else:
        raise RuntimeError(f"run `flow init` to init flow client config first")
    return ip, http_port, api_version


default_ip, default_port, default_version = parse_config()


class FlowClient(BaseFlowClient):
    job = api.Job()
    component = api.Component()
    data = api.Data()
    queue = api.Queue()
    table = api.Table()
    task = api.Task()
    model = api.Model()
    tag = api.Tag()
    priviledge = api.Priviledge()

    def __init__(self, ip=None, port=None, version=None):
        if ip is None or port is None or version is None:
            try:
                ip, port, version = parse_config()
            except Exception as e:
                raise RuntimeError(f"init FlowClient without ip/port/version provided, "
                                   f"and parse default settings failed") from e

        super().__init__(ip, port, version)
        self.API_BASE_URL = 'http://%s:%s/%s/' % (ip, port, version)
