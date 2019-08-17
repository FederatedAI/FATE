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
scene_key_separator = '_'


def check_scene_info(scene_id, role, party_id, all_party):
    return scene_id and role and party_id


def gen_scene_key(scene_id, role, party_id, all_party):
    return scene_key_separator.join([str(scene_id), role if role else 'all', str(party_id), join_all_party(all_party)])


def join_all_party(all_party):
    """
    Join all party as party key
    :param all_party:
        "role": {
            "guest": [9999],
            "host": [10000],
            "arbiter": [10000]
         }
    :return:
    """
    if not all_party:
        all_party_key = 'all'
    elif isinstance(all_party, dict):
        sorted_role_name = sorted(all_party.keys())
        all_party_key = '-'.join(['|'.join([str(p) for p in sorted(set(all_party[role_name]))])
                                  for role_name in sorted_role_name])
    else:
        all_party_key = None
    return all_party_key



