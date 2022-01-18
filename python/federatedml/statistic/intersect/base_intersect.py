#
#  Copyright 2021 The FATE Authors. All Rights Reserved.
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

import functools
import uuid

from federatedml.param.intersect_param import IntersectParam
from federatedml.statistic.intersect.intersect_preprocess import BitArray
from federatedml.transfer_variable.transfer_class.intersection_func_transfer_variable import IntersectionFuncTransferVariable
from federatedml.util import LOGGER


class Intersect(object):
    def __init__(self):
        super().__init__()
        self.cache_id = None
        self.model_param = IntersectParam()
        self.transfer_variable = None
        self.cache_transfer_variable = IntersectionFuncTransferVariable().cache_id_from_host
        self.filter = None
        self.intersect_num = None
        self.cache = None
        self.model_param_name = "IntersectModelParam"
        self.model_meta_name = "IntersectModelMeta"

        self._guest_id = None
        self._host_id = None
        self._host_id_list = None

    def load_params(self, param):
        self.model_param = param
        self.intersect_method = param.intersect_method
        self.only_output_key = param.only_output_key
        self.sync_intersect_ids = param.sync_intersect_ids
        self.cardinality_only = param.cardinality_only
        self.sync_cardinality = param.sync_cardinality
        self.run_preprocess = param.run_preprocess
        self.intersect_preprocess_params = param.intersect_preprocess_params
        self.run_cache = param.run_cache

    @property
    def guest_party_id(self):
        return self._guest_id

    @guest_party_id.setter
    def guest_party_id(self, guest_id):
        if not isinstance(guest_id, int):
            raise ValueError("party id should be integer, but get {}".format(guest_id))
        self._guest_id = guest_id

    @property
    def host_party_id(self):
        return self._host_id

    @host_party_id.setter
    def host_party_id(self, host_id):
        if not isinstance(host_id, int):
            raise ValueError("party id should be integer, but get {}".format(host_id))
        self._host_id = host_id

    @property
    def host_party_id_list(self):
        return self._host_id_list

    @host_party_id_list.setter
    def host_party_id_list(self, host_id_list):
        if not isinstance(host_id_list, list):
            raise ValueError(
                "type host_party_id should be list, but get {} with {}".format(type(host_id_list), host_id_list))
        self._host_id_list = host_id_list

    def get_intersect_method_meta(self):
        pass

    def get_intersect_key(self, party_id):
        pass

    def load_intersect_key(self, cache_meta):
        pass

    def run_intersect(self, data_instances):
        raise NotImplementedError("method should not be called here")

    def run_cardinality(self, data_instances):
        raise NotImplementedError("method should not be called here")

    def generate_cache(self, data_instances):
        raise NotImplementedError("method should not be called here")

    @staticmethod
    def extract_cache_list(cache_data, party_list):
        if not isinstance(party_list, list):
            party_list = [party_list]
        cache_list = [cache_data.get(str(party_id)) for party_id in party_list]
        return cache_list

    def run_cache_intersect(self, data_instances, cache_data):
        raise NotImplementedError("method should not be called here")

    def set_flowid(self, flowid=0):
        if self.transfer_variable is not None:
            self.transfer_variable.set_flowid(flowid)

    @staticmethod
    def get_value_from_data(intersect_ids, data_instances):
        if intersect_ids is not None:
            intersect_ids = intersect_ids.join(data_instances, lambda i, d: d)
            intersect_ids.schema = data_instances.schema
            LOGGER.info("obtain intersect data_instances!")

        return intersect_ids

    @staticmethod
    def get_common_intersection(intersect_ids_list: list, keep_encrypt_ids=False):
        if len(intersect_ids_list) == 1:
            return intersect_ids_list[0]

        if keep_encrypt_ids:
            def f(v_prev, v): return v_prev + v
        else:
            def f(v_prev, v): return "id"

        intersect_ids = None
        for i, value in enumerate(intersect_ids_list):
            if intersect_ids is None:
                intersect_ids = value
                continue
            intersect_ids = intersect_ids.join(value, f)

        return intersect_ids

    @staticmethod
    def extract_intersect_ids(intersect_ids, all_ids, keep_both=False):
        if keep_both:
            intersect_ids = intersect_ids.join(all_ids, lambda e, h: [e, h])
        else:
            intersect_ids = intersect_ids.join(all_ids, lambda e, h: h)
        return intersect_ids

    @staticmethod
    def filter_intersect_ids(encrypt_intersect_ids, keep_encrypt_ids=False):
        if keep_encrypt_ids:
            def f(k, v): return (v, [k])
        else:
            def f(k, v): return (v, 1)
        if len(encrypt_intersect_ids) > 1:
            raw_intersect_ids = [e.map(f) for e in encrypt_intersect_ids]
            intersect_ids = Intersect.get_common_intersection(raw_intersect_ids, keep_encrypt_ids)
        else:
            intersect_ids = encrypt_intersect_ids[0]
            intersect_ids = intersect_ids.map(f)
        return intersect_ids

    @staticmethod
    def map_raw_id_to_encrypt_id(raw_id_data, encrypt_id_data, keep_value=False):
        encrypt_id_data_exchange_kv = encrypt_id_data.map(lambda k, v: (v, k))
        encrypt_raw_id = raw_id_data.join(encrypt_id_data_exchange_kv, lambda r, e: (e, r))
        if keep_value:
            encrypt_common_id = encrypt_raw_id.map(lambda k, v: (v[0], v[1]))
        else:
            encrypt_common_id = encrypt_raw_id.map(lambda k, v: (v[0], "id"))

        return encrypt_common_id

    @staticmethod
    def map_encrypt_id_to_raw_id(encrypt_id_data, raw_id_data, keep_encrypt_id=True):
        """

        Parameters
        ----------
        encrypt_id_data: E(id)
        raw_id_data: (E(id), (id, v))
        keep_encrypt_id: bool

        Returns
        -------
        (id, E(id))

        """
        encrypt_id_raw_id = raw_id_data.join(encrypt_id_data, lambda r, e: r)
        if keep_encrypt_id:
            raw_id = encrypt_id_raw_id.map(lambda k, v: (v[0], k))
        else:
            raw_id = encrypt_id_raw_id.map(lambda k, v: (v[0], 1))
        return raw_id

    @staticmethod
    def hash(value, hash_operator, salt=''):
        h_value = hash_operator.compute(value, suffix_salt=salt)
        return h_value

    @staticmethod
    def generate_new_uuid():
        return str(uuid.uuid4())

    @staticmethod
    def insert_key(kv_iterator, filter, hash_operator=None, salt=None):
        res_filter = None
        for k, _ in kv_iterator:
            if hash_operator:
                res_filter = filter.insert(hash_operator.compute(k, suffix_salt=salt))
            else:
                res_filter = filter.insert(k)
        return res_filter

    @staticmethod
    def count_key_in_filter(kv_iterator, filter):
        count = 0
        for k, _ in kv_iterator:
            count += filter.check(k)
        return count

    @staticmethod
    def construct_filter(data, false_positive_rate, hash_method, random_state, hash_operator=None, salt=None):
        n = data.count()
        m, k = BitArray.get_filter_param(n, false_positive_rate)
        filter = BitArray(m, k, hash_method, random_state)
        LOGGER.debug(f"filter bit count is: {filter.bit_count}")
        LOGGER.debug(f"filter hash func count: {filter.hash_func_count}")
        f = functools.partial(Intersect.insert_key, filter=filter, hash_operator=hash_operator, salt=salt)
        new_array = data.mapPartitions(f).reduce(lambda x, y: x | y)
        LOGGER.debug(f"filter array obtained")
        filter.set_array(new_array)
        # LOGGER.debug(f"after insert, filter sparsity is: {filter.sparsity}")
        return filter
