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
from collections import defaultdict

from arch.api.utils import log_utils
LOGGER = log_utils.getLogger()

class RepeatedIDIntersect(object):
    def __init__(self, repeated_id_owner:bool):
        self.repeated_id_owner = repeated_id_owner

    def generate_id_map(self, data):
        if not self.repeated_id_owner:
           LOGGER.warning("Not a repeated id owner, will not generate id map")
           return None

        data = data.mapValues(lambda v: v[0])

        local_data = data.collect()
        all_id_map = defaultdict(list)
        final_id_map = {}

        for _data in local_data:
            all_id_map[_data[1]].append(_data[0])

        for k, v in all_id_map.items():
            if len(v) >= 2:
                final_id_map[k] = v

        return final_id_map

    @staticmethod
    def func_restructure_id(k, v, id_map:dict):
        result = []
        if id_map.get(k) is not None:
            for new_id in id_map[k]:
                result.append((new_id, v))

        return result

    def restructure_new_data(self, data, id_map):
        new_data = data.flatMap(data, id_map)
        repeared_ids = [ k for k, _ in id_map.items()]











