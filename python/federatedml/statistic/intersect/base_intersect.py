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
import numpy as np
import uuid

from federatedml.param.intersect_param import IntersectParam, IntersectPreProcessParam
from federatedml.protobuf.generated.intersect_meta_pb2 import IntersectModelMeta, IntersectPreProcessMeta
from federatedml.protobuf.generated.intersect_param_pb2 import Filter, IntersectModelParam, RSAKey, PHKey
from federatedml.statistic.intersect.intersect_preprocess import BitArray
from federatedml.util import LOGGER, consts

class Intersect(object):
    def __init__(self):
        super().__init__()
        self.model_param = IntersectParam()
        self.transfer_variable = None
        self.filter = None
        self.intersect_num = None
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
        return IntersectModelMeta()

    def get_intersect_key(self):
        pass

    def load_intersect_key(self, intersect_key):
        pass

    def _get_meta(self):
        preprocess_params = IntersectPreProcessMeta(
            false_positive_rate=self.intersect_preprocess_params.false_positive_rate,
            encrypt_method=self.intersect_preprocess_params.encrypt_method,
            hash_method=self.intersect_preprocess_params.hash_method,
            random_state=self.intersect_preprocess_params.random_state)
        if self.intersect_method == consts.RSA:
            meta_obj = IntersectModelMeta(intersect_method=self.intersect_method,
                                          intersect_preprocess_params=preprocess_params,
                                          rsa_params=self.get_intersect_method_meta())
        elif self.intersect_method == consts.PH:
            meta_obj = IntersectModelMeta(intersect_method=self.intersect_method,
                                          intersect_preprocess_params=preprocess_params,
                                          ph_params=self.get_intersect_method_meta())
        else:
            meta_obj = IntersectModelMeta(intersect_method=self.intersect_method,
                                          intersect_preprocess_params=preprocess_params)
        return meta_obj

    def _get_param(self):
        param_obj = IntersectModelParam()
        if self.cardinality_only:
            intersect_filter = Filter()
            rsa_encrypt_key = RSAKey()
            ph_encrypt_key = PHKey()
            if self.filter:
                intersect_filter = Filter(bit_count=self.filter.bit_count,
                                          filter_array=self.filter.get_array(),
                                          filter_id=self.filter.id,
                                          salt=self.filter.salt,
                                          hash_func_count=self.filter.hash_func_count)
            if self.intersect_method == consts.RSA:
                rsa_encrypt_key = self.get_intersect_key()
            param_obj = IntersectModelParam(intersect_filter=intersect_filter,
                                            rsa_encrypt_key=rsa_encrypt_key,
                                            ph_encrypt_key=ph_encrypt_key)

        return param_obj

    def get_model(self):
        meta_obj = self._get_meta()
        param_obj = self._get_param()

        result = {
            self.model_meta_name: meta_obj,
            self.model_param_name: param_obj
        }
        return result

    def load_model(self, model_dict):
        meta_obj, param_obj = model_dict[self.model_meta_name], model_dict[self.model_param_name]
        proprocess_params = meta_obj.intersect_preprocess_params
        self.intersect_preprocess_params = IntersectPreProcessParam(**proprocess_params)
        # self.intersect_preprocess_params.false_positive_rate = meta_obj.false_positive_rate
        # self.intersect_preprocess_params.hash_method = meta_obj.hash_method
        # self.intersect_preprocess_params.random_state = meta_obj.random_state
        if param_obj.filter and param_obj.filter.bit_count > 0:
            filter_obj = param_obj.filter
            filter_array = np.array(filter_obj.filter_array)
            self.filter = BitArray(bit_count=filter_obj.bit_count,
                                   hash_func_count=filter_obj.hash_func_count,
                                   hash_method=proprocess_params.hash_method,
                                   random_state=proprocess_params.random_state,
                                   salt=list(filter_obj.salt))
            self.filter.id = filter_obj.id
            self.filter.set_array(filter_array)
        if meta_obj.intersect_method == consts.RSA:
            self.load_intersect_key(param_obj.rsa_encrypt_key)

    def run_intersect(self, data_instances):
        raise NotImplementedError("method should not be called here")

    def run_cardinality(self, data_instances):
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
            f = lambda id, v: id + v
        else:
            f = lambda id, v: "id"

        intersect_ids = None
        for i, value in enumerate(intersect_ids_list):
            if intersect_ids is None:
                intersect_ids = value
                continue
            intersect_ids = intersect_ids.join(value, f)

        return intersect_ids

    @staticmethod
    def extract_intersect_ids(intersect_ids, all_ids):
        intersect_ids = intersect_ids.join(all_ids, lambda e, h: h)
        return intersect_ids

    @staticmethod
    def filter_intersect_ids(encrypt_intersect_ids, keep_encrypt_ids=False):
        if keep_encrypt_ids:
            f = lambda k, v: (v, [k])
        else:
            f = lambda k, v: (v, 1)
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
    def map_encrypt_id_to_raw_id(encrypt_id_data, raw_id_data):
        """

        Parameters
        ----------
        encrypt_id_data: E(id)
        raw_id_data: (E(id), (id, v))

        Returns
        -------
        (id, E(id))

        """
        encrypt_id_raw_id = raw_id_data.join(encrypt_id_data, lambda r, e: r)
        raw_id = encrypt_id_raw_id.map(lambda k, v: (v[0], k))
        return raw_id

    @staticmethod
    def hash(value, hash_operator, salt=''):
        h_value = hash_operator.compute(value, suffix_salt=salt)
        return h_value

    @staticmethod
    def generate_new_uuid():
        return str(uuid.uuid1())

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
        f = functools.partial(Intersect.insert_key, filter=filter, hash_operator=hash_operator, salt=salt)
        # ind_list = data.map(lambda k, v: (filter.get_ind_set(k), k))
        # ind_list.mapPartitions(f, use_previous_behavior=False)
        new_array = data.mapPartitions(f).reduce(lambda x, y: x | y)
        LOGGER.debug(f"filter array obtained")
        filter.set_array(new_array)
        # LOGGER.debug(f"after insert, filter sparsity is: {filter.sparsity}")
        return filter
