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
from urllib import parse

from kazoo.client import KazooClient

from arch.api.utils import file_utils
from fate_flow.entity.runtime_config import RuntimeConfig


class CenterConfig(object):
    USE_CONFIGURATION_CENTER = False
    SERVERS = None
    SERVINGS_ZK_PATH = None

    @staticmethod
    def get_settings(path, servings_zk_path=None):
        if servings_zk_path and CenterConfig.USE_CONFIGURATION_CENTER:
            return CenterConfig.get_servings_from_zookeeper(servings_zk_path)
        return CenterConfig.get_settings_from_file(path)

    @staticmethod
    def get_settings_from_file(path):
        server_conf = file_utils.load_json_conf("arch/conf/server_conf.json")
        data = server_conf
        for k in path.split('/')[1:]:
            data = data.get(k, None)
        return data

    @staticmethod
    def update_servings(event):
        nodes = RuntimeConfig.ZK.get_children(CenterConfig.SERVINGS_ZK_PATH)
        CenterConfig.SERVERS = nodes_unquote(nodes)

    @staticmethod
    def get_servings_from_zookeeper(path):
        try:
            zk = RuntimeConfig.ZK
            nodes = zk.get_children(path, watch=CenterConfig.update_servings)
            return nodes_unquote(nodes)
        except Exception as e:
            raise Exception('loading servings node  failed from zookeeper: {}'.format(e))

    @staticmethod
    def init(hosts, use_configuation_center, servings_zk_path):
        if not use_configuation_center:
            CenterConfig.SERVERS = CenterConfig.get_settings('/servers/servings')
        else:
            zk = KazooClient(hosts=hosts)
            zk.start()
            RuntimeConfig.init_config(ZK=zk)
            CenterConfig.USE_CONFIGURATION_CENTER = True
            CenterConfig.SERVINGS_ZK_PATH = servings_zk_path
            CenterConfig.SERVERS = CenterConfig.get_servings_from_zookeeper(servings_zk_path)


def nodes_unquote(nodes):
    urls = [parse.unquote(node) for node in nodes]
    servings = []
    for url in urls:
        try:
            servings.append(url.split('/')[2])
        except:
            pass
    return servings



