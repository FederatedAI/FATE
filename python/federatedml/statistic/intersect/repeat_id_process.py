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

from fate_arch.session import computing_session as session
from federatedml.feature.instance import Instance
from federatedml.transfer_variable.transfer_class.repeated_id_intersect_transfer_variable import \
    RepeatedIDIntersectTransferVariable
from federatedml.util import consts
from federatedml.util import LOGGER


class RepeatedIDIntersect(object):
    """
    This will support repeated ID intersection using ID expanding.
    """

    def __init__(self, repeated_id_owner: str, role: str):
        self.repeated_id_owner = repeated_id_owner
        self.transfer_variable = RepeatedIDIntersectTransferVariable()
        self.role = role
        self.id_map = None
        self.version = None
        self.owner_src_data = None
        self.data_type = None

    def __get_data_type(self, data):
        if self.data_type is None:
            one_feature = data.first()
            if isinstance(one_feature[1], Instance):
                self.data_type = Instance
            else:
                self.data_type = list

        return self.data_type

    @staticmethod
    def __to_id_map(data):
        id_map = defaultdict(list)
        for d in data:
            idx = d[1].features[0] if isinstance(d[1], Instance) else d[1][0]
            id_map[idx].append(d[0])

        return [(k, v) for k, v in id_map.items()]

    @staticmethod
    def __reduce_id_map(x1, x2):
        return x1 + x2

    def __generate_id_map(self, data):
        if self.role != self.repeated_id_owner:
            LOGGER.warning("Not a repeated id owner, will not generate id map")
            return

        id_map = data.mapReducePartitions(self.__to_id_map, self.__reduce_id_map)
        id_map.filter(lambda k, v: len(v) >= 2)

        return id_map

    # def __generate_id_map(self, data):
    #     if self.role != self.repeated_id_owner :
    #         LOGGER.warning("Not a repeated id owner, will not generate id map")
    #         return
    #
    #     data_type = self.__get_data_type(data)
    #     if isinstance(data_type, Instance):
    #         data = data.mapValues(lambda v: v.features[0])
    #     else:
    #         data = data.mapValues(lambda v: v[0])
    #
    #     local_data = data.collect()
    #     all_id_map = defaultdict(list)
    #     final_id_map = {}
    #
    #     for _data in local_data:
    #         all_id_map[str(_data[1])].append(_data[0])
    #
    #     for k, v in all_id_map.items():
    #         if len(v) >= 2:
    #             final_id_map[k] = v
    #
    #     return final_id_map

    @staticmethod
    def __func_restructure_id(k, id_map: list):
        return [(new_id, k) for new_id in id_map]

    # @staticmethod
    # def __func_restructure_id(k, v):
    #     data, id_map = v[0], v[1]
    #     result = [(new_id, data) for new_id in id_map]
    #     return result

    @staticmethod
    def __func_restructure_id_for_partner(k, v):
        data, id_map = v[0], v[1]
        return [(new_id, data) for new_id in id_map]

    @staticmethod
    def __func_restructure_instance(v):
        features = [v.features[0]]
        if len(v.features) > 2:
            features += v.features[2:]

        v.features = features
        return v

    def __restructure_owner_sample_ids(self, data, id_map):
        if self.version == "1.6.0":
            rids = id_map.flatMap(functools.partial(self.__func_restructure_id))
            _data = data.union(rids, lambda dv, rv: dv)

            if isinstance(self.__get_data_type(data), Instance):
                r_data = self.owner_src_data.join(_data, lambda ov, dv: self.__func_restructure_instance(ov))
            else:
                r_data = self.owner_src_data.join(_data, lambda ov, dv: ov[0] + ov[2:])

        return r_data

    def __restructure_partner_sample_ids(self, data, id_map):
        _data = data.join(id_map, lambda dv, iv: (dv, iv))
        repeated_ids = id_map.flatMap(functools.partial(self.__func_restructure_id_for_partner))
        sub_data = data.subtract_by_key(id_map)
        return sub_data.union(repeated_ids, lambda sv, rv: sv)

    def __restructure_sample_ids(self, data, id_map):
        if self.role == self.repeated_id_owner:
            return self.__restructure_owner_sample_ids(data, id_map)
        else:
            return self.__restructure_partner_sample_ids(data, id_map)

    def recover(self, data):
        LOGGER.info("Start repeated id processing.")
        if self.role != self.repeated_id_owner:
            LOGGER.info("Not repeated_id_owner, return!")

        self.id_map = self.__generate_id_map(data)

        # original_schema = data.schema
        if self.__get_data_type(data) == Instance:
            data = data.mapValues(
                lambda v: Instance(features=np.array(v.features[1:], dtype=np.float), label=v.label,
                                   inst_id=v.inst_id, weight=v.weight))
        else:
            data = data.mapValues(lambda v: v[1:])
        # data.schema = original_schema

        if data.schema.get('header') is not None:
            data.schema['header'] = data.schema['header'][1:]

        LOGGER.info("Finish recover real ids")
        return data

    def expand(self, data):
        if self.repeated_id_owner == consts.HOST:
            id_map_federation = self.transfer_variable.id_map_from_host
            partner_role = consts.GUEST
        else:
            id_map_federation = self.transfer_variable.id_map_from_guest
            partner_role = consts.HOST

        if self.repeated_id_owner == self.role:
            self.id_map = self.id_map.join(data, lambda i, d: i)
            LOGGER.info("Find repeated id_map from intersection ids")

            id_map_federation.remote(self.id_map,
                                     role=partner_role,
                                     idx=-1)
            LOGGER.info("Remote id_map to partner")
        else:
            # original_schema = data.schema
            id_map = id_map_federation.get(idx=0)
            LOGGER.info("Get id_map from owner.")
            # data.schema = original_schema

        return self.__restructure_sample_ids(data, self.id_map)
