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

from federatedml.secureprotol.elliptic_curve_encryption import EllipticCurve
from federatedml.secureprotol.hash.hash_factory import Hash
from federatedml.statistic.intersect.base_intersect import Intersect
from federatedml.transfer_variable.transfer_class.ecdh_intersect_transfer_variable import EcdhIntersectTransferVariable
from federatedml.util import LOGGER, consts


class EcdhIntersect(Intersect):
    """
    adapted from Secure Information Retrieval Module
    """

    def __init__(self):
        super().__init__()
        self.role = None
        self.transfer_variable = EcdhIntersectTransferVariable()
        self.curve_instance = None

    def load_params(self, param):
        super().load_params(param=param)
        if len(self.host_party_id_list) > 1:
            raise ValueError(f"ECDH method currently only support single-host tasks.")
        self.ecdh_params = param.ecdh_params
        self.hash_operator = Hash(param.ecdh_params.hash_method, hex_output=False)
        self.salt = self.ecdh_params.salt
        self.curve = self.ecdh_params.curve

    def get_intersect_method_meta(self):
        ecdh_meta = {"intersect_method": consts.ECDH,
                     "hash_method": self.ecdh_params.hash_method,
                     "salt": self.salt,
                     "curve": self.curve}
        return ecdh_meta

    def init_curve(self, curve_key=None):
        self.curve_instance = EllipticCurve(self.curve, curve_key)

    @staticmethod
    def get_mode(reserve_original_key=False, reserve_original_value=False):
        if reserve_original_key and reserve_original_value:
            return 5
        if reserve_original_key:
            return 4
        if reserve_original_value:
            return 3
        return 1

    @staticmethod
    def _encrypt_id(data_instances, curve_instance, reserve_original_key=False, hash_operator=None, salt='',
                    reserve_original_value=False):
        """
        Encrypt the key (ID) of input Table
        :param curve: curve object
        :param data_instance: Table
        :param reserve_original_key: (enc_key, ori_key) if reserve_original_key == True, otherwise (enc_key, -1)
        :param hash_operator: if provided, use map_hash_encrypt
        :param salt: if provided, use for map_hash_encrypt
        : param reserve_original_value:
            (enc_key, (ori_key, val)) for reserve_original_key == True and reserve_original_value==True;
            (ori_key, (enc_key, val)) for only reserve_original_value == True.
        :return:
        """
        mode = EcdhIntersect.get_mode(reserve_original_key, reserve_original_value)
        if hash_operator is not None:
            return curve_instance.map_hash_encrypt(data_instances, mode=mode, hash_operator=hash_operator, salt=salt)
        return curve_instance.map_encrypt(data_instances, mode=mode)

    @staticmethod
    def _sign_id(data_instances, curve_instance, reserve_original_key=False, reserve_original_value=False):
        """
        Encrypt the key (ID) of input Table
        :param curve_instance: curve object
        :param data_instance: Table
        :param reserve_original_key: (enc_key, ori_key) if reserve_original_key == True, otherwise (enc_key, -1)
        : param reserve_original_value:
            (enc_key, (ori_key, val)) for reserve_original_key == True and reserve_original_value==True;
            (ori_key, (enc_key, val)) for only reserve_original_value == True.
        :return:
        """
        mode = EcdhIntersect.get_mode(reserve_original_key, reserve_original_value)
        return curve_instance.map_sign(data_instances, mode=mode)

    def _exchange_id(self, id, replace_val=True):
        """
        :param id: Table in the form (id, 0)
        :return:
        """
        pass

    def _sync_doubly_encrypted_id(self, id):
        """
        host -> guest
        :param id:
        :return:
        """
        pass

    def get_intersect_doubly_encrypted_id(self, data_instances, keep_key=True):
        raise NotImplementedError("This method should not be called here")

    def decrypt_intersect_doubly_encrypted_id(self, id_intersect_cipher_cipher):
        raise NotImplementedError("This method should not be called here")

    def get_intersect_doubly_encrypted_id_from_cache(self, data_instances, cache_set):
        raise NotImplementedError("This method should not be called here")

    def run_intersect(self, data_instances):
        if len(self.host_party_id_list) > 1:
            raise ValueError(f"Intersection with ECDH only supports single-host task.")
        LOGGER.info("Start ECDH Intersection")
        id_intersect_cipher_cipher = self.get_intersect_doubly_encrypted_id(data_instances)
        intersect_ids = self.decrypt_intersect_doubly_encrypted_id(id_intersect_cipher_cipher)
        return intersect_ids

    def run_cache_intersect(self, data_instances, cache_data):
        if len(self.host_party_id_list) > 1:
            raise ValueError(f"Intersection with ECDH only supports single-host task.")
        LOGGER.info("Start ECDH Intersection with cache")
        id_intersect_cipher_cipher = self.get_intersect_doubly_encrypted_id_from_cache(data_instances, cache_data)
        intersect_ids = self.decrypt_intersect_doubly_encrypted_id(id_intersect_cipher_cipher)
        return intersect_ids
