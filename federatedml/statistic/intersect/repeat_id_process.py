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
import functools
import numpy as np

from arch.api import session
from arch.api.utils import log_utils
from federatedml.feature.instance import Instance
from federatedml.transfer_variable.transfer_class.repeated_id_intersect_transfer_variable import \
    RepeatedIDIntersectTransferVariable
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class RepeatedIDIntersect(object):
    """
    This will support repeated ID intersection using ID expanding.
    """

    def __init__(self, repeated_id_owner: str, role: str):
        self.repeated_id_owner = repeated_id_owner
        self.transfer_variable = RepeatedIDIntersectTransferVariable()
        self.role = role

    def __generate_id_map(self, data: session.table) -> dict:
        if not self.repeated_id_owner:
            LOGGER.warning("Not a repeated id owner, will not generate id map")
            return {}

        one_feature = data.first()
        if isinstance(one_feature[1], Instance):
            data = data.mapValues(lambda v: v.features[0])
        else:
            data = data.mapValues(lambda v: v[0])

        local_data = data.collect()
        all_id_map = defaultdict(list)
        final_id_map = {}

        for _data in local_data:
            all_id_map[str(_data[1])].append(_data[0])

        for k, v in all_id_map.items():
            if len(v) >= 2:
                final_id_map[k] = v

        return final_id_map

    @staticmethod
    def __func_restructure_id(k, v, id_map: dict):
        if id_map.get(k) is not None:
            result = []
            for new_id in id_map[k]:
                result.append((new_id, v))
            return result

        return [(k, v)]

    def run(self, data):
        LOGGER.info("Start repeated id processing.")
        id_map_federation = self.transfer_variable.id_map_from_guest
        party_role = consts.HOST
        if self.repeated_id_owner == consts.HOST:
            id_map_federation = self.transfer_variable.id_map_from_host
            party_role = consts.GUEST

        LOGGER.info("repeated_id_owner:{}".format(self.repeated_id_owner))

        original_schema = data.schema
        if self.repeated_id_owner == self.role:
            id_map = self.__generate_id_map(data)
            LOGGER.info("finish generate id_map, id_map:{}".format(id_map))

            id_map_federation.remote(id_map,
                                     role=party_role,
                                     idx=-1)

            one_feature = data.first()
            if isinstance(one_feature[1], Instance):
                data = data.mapValues(
                    lambda v: Instance(features=np.array(v.features[1:], dtype=np.float), label=v.label,
                                       inst_id=v.inst_id, weight=v.weight))
            else:
                data = data.mapValues(lambda v: v[1:])
            data.schema = original_schema
            if data.schema.get('header') is not None:
                data.schema['header'] = data.schema['header'][1:]
        else:
            id_map = id_map_federation.get(idx=0)
            LOGGER.info("Get id_map from owner.")
            data = data.flatMap(functools.partial(self.__func_restructure_id, id_map=id_map))
            data.schema = original_schema

        LOGGER.info("Finish repeated id process for owner")

        return data
