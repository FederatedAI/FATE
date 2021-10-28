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

from federatedml.feature.instance import Instance
from federatedml.transfer_variable.transfer_class.match_id_intersect_transfer_variable import \
    MatchIDIntersectTransferVariable
from federatedml.util import consts
from federatedml.util import LOGGER


class MatchIDIntersect(object):
    """
    This will support repeated ID intersection using ID expanding.
    """

    def __init__(self, sample_id_generator: str, role: str):
        self.sample_id_generator = sample_id_generator
        self.transfer_variable = MatchIDIntersectTransferVariable()
        self.role = role
        self.id_map = None
        self.version = None
        self.owner_src_data = None
        self.data_type = None
        self.with_sample_id = False

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

    @staticmethod
    def __to_sample_id_map(data):
        id_map = defaultdict(list)
        for d in data:
            id_map[d[1].inst_id].append(d[0])

        return [(k, v) for k, v in id_map.items()]

    def __generate_id_map(self, data):
        if self.role != self.sample_id_generator:
            LOGGER.warning("Not a repeated id owner, will not generate id map")
            return

        if not self.with_sample_id:
            all_id_map = data.mapReducePartitions(self.__to_id_map, self.__reduce_id_map)
            id_map = all_id_map.filter(lambda k, v: len(v) >= 2)
        else:
            id_map = data.mapReducePartitions(self.__to_sample_id_map, self.__reduce_id_map)

        return id_map

    @staticmethod
    def __func_restructure_id(k, id_map: list):
        return [(new_id, k) for new_id in id_map]

    @staticmethod
    def __func_restructure_id_for_partner(k, v):
        data, id_map = v[0], v[1]
        return [(new_id, data) for new_id in id_map]

    @staticmethod
    def __func_restructure_sample_id_for_partner(k, v):
        data, id_map = v[0], v[1]
        return [(new_id, data) for new_id in id_map]

    @staticmethod
    def __func_restructure_instance(v):
        v.features = v.features[1:]
        return v

    def __restructure_owner_sample_ids(self, data, id_map):
        rids = id_map.flatMap(functools.partial(self.__func_restructure_id))
        if not self.with_sample_id:
            _data = data.union(rids, lambda dv, rv: dv)

            if self.__get_data_type(self.owner_src_data) == Instance:
                r_data = self.owner_src_data.join(_data, lambda ov, dv: self.__func_restructure_instance(ov))
            else:
                r_data = self.owner_src_data.join(_data, lambda ov, dv: ov[1:])

            r_data.schema = self.owner_src_data.schema
            if r_data.schema.get('header') is not None:
                r_data.schema['header'] = r_data.schema['header'][1:]
        else:
            r_data = self.owner_src_data.join(rids, lambda ov, dv: ov)
            r_data.schema = self.owner_src_data.schema

        return r_data

    def __restructure_partner_sample_ids(self, data, id_map, match_data=None):
        data = data.join(match_data, lambda k, v: v)
        _data = data.join(id_map, lambda dv, iv: (dv, iv))
        # LOGGER.debug(f"_data is: {_data.first()}")
        repeated_ids = _data.flatMap(functools.partial(self.__func_restructure_id_for_partner))
        # LOGGER.debug(f"restructure id for partner called, result is: {repeated_ids.first()}")
        if not self.with_sample_id:
            sub_data = data.subtractByKey(id_map)
            expand_data = sub_data.union(repeated_ids, lambda sv, rv: sv)
        else:
            expand_data = repeated_ids

        expand_data.schema = data.schema
        if match_data:
            expand_data.schema = match_data.schema

        return expand_data

    def __restructure_sample_ids(self, data, id_map, match_data=None):
        # LOGGER.debug(f"id map is: {self.id_map.first()}")
        if self.role == self.sample_id_generator:
            return self.__restructure_owner_sample_ids(data, id_map)
        else:
            return self.__restructure_partner_sample_ids(data, id_map, match_data)

    def generate_intersect_data(self, data):
        if self.__get_data_type(data) == Instance:
            if not self.with_sample_id:
                _data = data.map(
                    lambda k, v: (v.features[0], 1))
            else:
                _data = data.map(lambda k, v: (v.inst_id, v))
        else:
            _data = data.mapValues(lambda k, v: (v[0], 1))

        _data.schema = data.schema
        LOGGER.info("Finish recover real ids")

        return _data

    def use_sample_id(self):
        self.with_sample_id = True

    def recover(self, data):
        LOGGER.info("Start repeated id processing.")

        if self.role == self.sample_id_generator:
            LOGGER.info("Start to generate id_map")
            self.id_map = self.__generate_id_map(data)
            self.owner_src_data = data
        else:
            if not self.with_sample_id:
                LOGGER.info("Not sample_id_generator, return!")
                return data

        return self.generate_intersect_data(data)

    def expand(self, data, owner_only=False, match_data=None):
        if self.sample_id_generator == consts.HOST:
            id_map_federation = self.transfer_variable.id_map_from_host
            partner_role = consts.GUEST
        else:
            id_map_federation = self.transfer_variable.id_map_from_guest
            partner_role = consts.HOST

        if self.sample_id_generator == self.role:
            self.id_map = self.id_map.join(data, lambda i, d: i)
            LOGGER.info("Find repeated id_map from intersection ids")
            if not owner_only:
                id_map_federation.remote(self.id_map,
                                         role=partner_role,
                                         idx=-1)
                LOGGER.info("Remote id_map to partner")
        else:
            if owner_only:
                return data
            self.id_map = id_map_federation.get(idx=0)
            LOGGER.info("Get id_map from owner.")

        return self.__restructure_sample_ids(data, self.id_map, match_data)
