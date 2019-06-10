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
from eggroll.api.version_control.control import get_latest_commit, get_id_library_table_name
from eggroll.api.utils.scene_utils import gen_scene_key, check_scene_info
from eggroll.api.utils.core import get_commit_id


def get_table_info(config, create=False):
    table_name, namespace, scene_id, role, party_id, all_party, data_type = config.get('table_name'), \
                                                                      config.get('namespace'), \
                                                                      config.get('scene_id'), \
                                                                      config.get('local', {}).get('role'), \
                                                                      config.get('local', {}).get('party_id'), \
                                                                      config.get('role'), \
                                                                      config.get('data_type')
    if not config.get('gen_table_info', False):
        return table_name, namespace
    if not namespace:
        if not check_scene_info(scene_id, role, party_id, all_party) or not data_type:
            return table_name, namespace
        namespace = get_scene_namespace(gen_scene_key(scene_id=scene_id,
                                                      role=role,
                                                      party_id=party_id,
                                                      all_party=all_party),
                                        data_type=data_type)
    if not table_name:
        if create:
            table_name = get_commit_id()
        else:
            table_name = get_latest_commit(data_table_namespace=namespace, branch='master')
    return table_name, namespace


def get_scene_namespace(scene_key, data_type):
    return '_'.join([scene_key, data_type])

