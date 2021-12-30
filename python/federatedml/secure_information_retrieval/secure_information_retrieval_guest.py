#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
import random

import numpy as np

from federatedml.feature.instance import Instance
from federatedml.secure_information_retrieval.base_secure_information_retrieval import \
    BaseSecureInformationRetrieval
from federatedml.param.sir_param import SecureInformationRetrievalParam
from federatedml.param.intersect_param import IntersectParam
from federatedml.secureprotol.oblivious_transfer.hauck_oblivious_transfer.hauck_oblivious_transfer_receiver import \
    HauckObliviousTransferReceiver
from federatedml.secureprotol.symmetric_encryption.py_aes_encryption import AESDecryptKey
from federatedml.secureprotol.symmetric_encryption.cryptor_executor import CryptoExecutor
from federatedml.statistic import data_overview
# from federatedml.secureprotol.symmetric_encryption.pohlig_hellman_encryption import PohligHellmanCipherKey
from federatedml.statistic.intersect import DhIntersectionGuest
from federatedml.util import consts, abnormal_detection, LOGGER


MODEL_PARAM_NAME = 'SecureInformationRetrievalParam'
MODEL_META_NAME = 'SecureInformationRetrievalMeta'


class SecureInformationRetrievalGuest(BaseSecureInformationRetrieval):
    def __init__(self):
        super(SecureInformationRetrievalGuest, self).__init__()
        self.oblivious_transfer = None
        self.target_block_index = None      # k-th block message is expected to obtain, with k in {0, 1, ..., N-1}

        # The following parameter restricts the range of the block number
        self.security_scale = np.log(500)   # block_num = 2 * exp(security_scale * security_level)

    def _init_model(self, param: SecureInformationRetrievalParam):
        self._init_base_model(param)
        self.intersection_obj = DhIntersectionGuest()
        self.intersection_obj.role = consts.GUEST
        intersect_param = IntersectParam(dh_params=self.dh_params)
        self.intersection_obj.load_params(intersect_param)
        self.intersection_obj.host_party_id_list = self.component_properties.host_party_idlist
        self.intersection_obj.guest_party_id = self.component_properties.guest_partyid

        if self.model_param.oblivious_transfer_protocol == consts.OT_HAUCK.lower():
            self.oblivious_transfer = HauckObliviousTransferReceiver()
        else:
            raise ValueError("SIR only supports Hauck's OT")

    def fit(self, data_inst):
        """

        :param data_inst: Table, only the key column of the Table is used
        :return:
        """
        abnormal_detection.empty_table_detection(data_inst)

        # 0. Raw retrieval
        match_data = data_inst
        self.with_inst_id = data_overview.check_with_inst_id(data_inst)
        if self.with_inst_id:
            match_data = self._recover_match_id(data_inst)

        if self.model_param.raw_retrieval or self.security_level == 0:
            LOGGER.info("enter raw information retrieval guest")
            # abnormal_detection.empty_table_detection(data_inst)
            data_output = self._raw_information_retrieval(match_data)
            self._display_result(block_num='N/A')
            if self.with_inst_id:
                data_output = self._restore_sample_id(data_output)
            data_output = self._compensate_set_difference(data_inst, data_output)
            return data_output

        # 1. Data pre-processing
        LOGGER.info("enter secure information retrieval guest")
        self.need_label = self._check_need_label()
        # abnormal_detection.empty_table_detection(data_inst)
        self._parse_security_level(match_data)
        if not self._check_oblivious_transfer_condition():
            self._failure_response()

        # 2. Find intersection
        id_list_intersect = self.intersection_obj.get_intersect_doubly_encrypted_id(match_data)[0]
        id_list_host_second_only = self.intersection_obj.id_list_remote_second[0]

        # 3. Send the re-indexed doubly encrypted ID to host
        self._fake_blocks(id_list_intersect, id_list_host_second_only)  # List[(EEi, -1)]
        LOGGER.info("faked blocks for obfuscation")

        # 4. Wait for host to restore value for the intersection
        LOGGER.info("waiting for host to restore interested values for the intersection")

        # 5. Execute OT as receiver
        LOGGER.info("enter oblivious transfer protocol as a receiver")
        target_key = self.oblivious_transfer.key_derivation(self.target_block_index)
        LOGGER.info("oblivious transfer key derived")

        # 6. Wait for host to encrypt and transmit, and then receive the encrypted interested values
        id_block_ciphertext, nonce = self._iteratively_get_encrypted_values()
        LOGGER.info("got encrypted interested values and nonce")
        target_block_cipher_id = self._non_committing_decrypt(
            id_block_ciphertext, nonce, target_key)  # (Eright, val)
        LOGGER.info("used the right key to decrypt the wanted values")

        # 7. Get (EEright, instance)
        target_block_cipher_cipher_id = self.intersection_obj.map_raw_id_to_encrypt_id(target_block_cipher_id,
                                                                                       id_list_host_second_only,
                                                                                       keep_value=True)
        # 8. Get (EEright, Eright_guest)
        id_list_local_first = self.intersection_obj.id_list_local_first[0]  # (Eright_guest, id)
        id_list_local_second = self.intersection_obj.id_list_local_second[0]  # (EEright, Eright_guest)

        # 9. Merge result
        # (Eright_guest, instance)
        id_list_cipher = self._merge_instance(target_block_cipher_cipher_id, id_list_local_second, self.need_label)
        data_output = self._merge(id_list_cipher, id_list_local_first)

        if self.with_inst_id:
            data_output = self._restore_sample_id(data_output)
        data_output = self._compensate_set_difference(data_inst, data_output)
        self._display_result()
        LOGGER.info("secure information retrieval finished")

        return data_output

    def _sync_nonce_list(self, nonce=None, time=0):
        nonce_list_result = self.transfer_variable.nonce_list.get(idx=0,
                                                                  suffix=(time,))
        LOGGER.info("Got {}-th nonce list from host".format(time))
        return nonce_list_result

    @staticmethod
    def _merge_instance(id_map1, id_map2, need_label):
        """

        :param id_map1: (a, b)
        :param id_map2: (a, c)
        :return: (c, b)
        """
        merge_table = id_map1.join(id_map2, lambda v, u: (u, v))
        if need_label:
            return merge_table.map(lambda k, v: (v[0], Instance(label=v[1], features=[])))
        else:
            return merge_table.map(lambda k, v: (v[0], Instance(features=v[1])))

    @staticmethod
    def _merge(id_map1, id_map2):
        """

        :param id_map1: (a, b)
        :param id_map2: (a, c)
        :return: (c, b)
        """
        merge_table = id_map1.join(id_map2, lambda v, u: (u, v))
        return merge_table.map(lambda k, v: (v[0], v[1]))

    def _composite_decrypt(self, id_list):
        """
        k, v -> k, Dv
        :param id_list:
        :return:
        """
        commutative_cipher = self.intersection_obj.commutative_cipher[0]
        return commutative_cipher.map_values_decrypt(id_list, mode=1)

    def _composite_encrypt(self, id_list):
        """
        k, v -> Ek, v
        :param id_list:
        :return:
        """
        commutative_cipher = self.intersection_obj.commutative_cipher[0]
        return commutative_cipher.map_encrypt(id_list, mode=2)

    def _decrypt_value(self, id_list):
        """

        :param id_list:
        :return:
        """

    def _non_committing_decrypt(self, id_block_ciphertext, nonce, target_key):
        """
        Use non-committing cipher to encrypt id blocks
        :param id_block_ciphertext: (Ei, Eval)
        :param nonce: bytes
        :param target_key: ObliviousTransferKey
        :return:
        """
        if self.model_param.non_committing_encryption == consts.AES.lower():
            aes_key = CryptoExecutor(AESDecryptKey(key=target_key.key, nonce=nonce))
        else:
            raise ValueError("only supports AES cipher for non-committing decryption")

        return aes_key.map_values_decrypt(id_block_ciphertext, mode=0)

    def _transmit_value_ciphertext(self, id_block=None, time=0):
        id_blocks = self.transfer_variable.id_blocks_ciphertext.get(idx=0,
                                                                    suffix=(time,))
        LOGGER.info("got {}-th id block ciphertext from host".format(time))
        return id_blocks

    def _decrypt_id_list(self, id_list):
        """

        :param id_list: (EEe, v)
        :return: (Ee, v)
        """
        commutative_cipher = self.intersection_obj.commutative_cipher[0]
        return commutative_cipher.map_decrypt(id_list, mode=2)

    def _sync_natural_indexation(self, id_list, time):
        self.transfer_variable.natural_indexation.remote(id_list,
                                                         suffix=(time,),
                                                         role=consts.HOST,
                                                         idx=0)
        LOGGER.info("sent naturally indexed block {} to host".format(time))

    def _fake_blocks(self, id_list_intersect, id_list_host, replacement=True):
        """
        Randomly sample self.block_num - 1 blocks with the same size as id_list_intersect from id_list_host
        :param id_list_intersect: Table in the form (intersect_ENC_id, -1)
        :param id_list_host: Table in the form (ENC_id, -1)
        :param replacement: bool
        :return: id_list_array: List[Table] with disjoint (ENC_id, -1) Tables
        """
        intersect_count = id_list_intersect.count()
        self.target_block_index = random.SystemRandom().randint(0, self.block_num - 1)
        for i in range(self.block_num):
            if i == self.target_block_index:
                id_block = id_list_intersect.join(id_list_host, lambda x, y: y)
            else:
                id_block = self.take_exact_sample(data_inst=id_list_host, exact_num=intersect_count)
                if not replacement:
                    id_list_host = id_list_host.subtractByKey(id_block)
            # id_block = self._decrypt_id_list(id_block)
            id_block = id_block.map(lambda k, v: (v, -1))
            self._sync_natural_indexation(id_block, time=i)

    @staticmethod
    def _id_list_array_indexation(id_list_array):
        """

        :param id_list_array: List(Table)
        :return:
        """
        for i in range(len(id_list_array)):
            id_list_array[i].mapValues(lambda v: i)
        return id_list_array

    def _parse_security_level(self, data_instance):
        # data_count_guest = data_instance.count()

        # block_num = 2 * exp(scale * level)
        self.block_num = int(np.ceil(2 * np.exp(self.security_scale * self.security_level)))
        LOGGER.info("parsed block num = {}".format(self.block_num))

        self._sync_block_num()

    def _raw_information_retrieval(self, data_instance):
        self.transfer_variable.raw_id_list.remote(data_instance.map(lambda k, v: (k, -1)),
                                                  role=consts.HOST,
                                                  idx=0)
        LOGGER.info("sent raw id list to host")

        data_output = self.transfer_variable.raw_value_list.get(idx=0)
        LOGGER.info("got raw value list from host")

        # data_output = self._compensate_set_difference(data_instance, data_output)

        return data_output

    @staticmethod
    def take_exact_sample(data_inst, exact_num):
        """
        Sample an exact number of instances from a Table
        :param data_inst: Table
        :param exact_num: int
        :return: Table
        """
        sample_inst = data_inst.sample(num=exact_num)
        return sample_inst

    def _sync_block_num(self):
        self.transfer_variable.block_num.remote(self.block_num,
                                                role=consts.HOST,
                                                idx=0)
        LOGGER.info("sent block num {} to host".format(self.block_num))

    def _compensate_set_difference(self, original_data, data_output):
        self.coverage = data_output.count() / original_data.count()
        import copy
        schema = copy.deepcopy(original_data.schema)
        if self.need_label:
            original_data = original_data.mapValues(lambda v: Instance(label="unretrieved", features=[],
                                                                       inst_id=v.inst_id))
        else:
            feature_count = len(self.target_cols)
            features = np.array(["unretrieved"] * feature_count)
            original_data = original_data.mapValues(lambda v: Instance(features=features,
                                                                       inst_id=v.inst_id))
        # LOGGER.debug(f"original data features is {list(original_data.collect())[0][1].features}")
        # LOGGER.debug(f"original data label is {list(original_data.collect())[0][1].label}")

        data_output = original_data.union(data_output, lambda v, u: u)
        # LOGGER.debug(f"data_output features after union is {list(data_output.collect())[0][1].features}")
        # LOGGER.debug(f"data_output label after union is {list(data_output.collect())[0][1].label}")
        if self.need_label:
            schema["label_name"] = "retrieved_value"
            schema["header"] = []
            data_output.schema = schema
        else:
            schema["label_name"] = None
            schema["header"] = self.target_cols
            data_output.schema = schema
        self._sync_coverage(original_data)
        return data_output

    def _sync_coverage(self, data_instance):
        coverage = self.coverage * data_instance.count()
        self.transfer_variable.coverage.remote(coverage,
                                               role=consts.HOST,
                                               idx=0)
        LOGGER.info(f"sent coverage {coverage} to host")

    def _iteratively_get_encrypted_values(self):
        """

        :return: Table, bytes
        """
        id_block_ciphertext = None
        nonce = None
        for i in range(self.block_num):
            id_block = self._transmit_value_ciphertext(time=i)     # List[(Ei, Eval)]
            nonce_inst = self._sync_nonce_list(time=i)

            if i != self.target_block_index:
                pass
            else:
                id_block_ciphertext = id_block
                nonce = nonce_inst

            self.transfer_variable.block_confirm.remote(True,
                                                        suffix=(i,),
                                                        role=consts.HOST,
                                                        idx=0)

        return id_block_ciphertext, nonce
