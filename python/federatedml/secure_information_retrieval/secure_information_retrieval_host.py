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

from federatedml.secure_information_retrieval.base_secure_information_retrieval import \
    BaseSecureInformationRetrieval, CryptoExecutor
from federatedml.param.sir_param import SecureInformationRetrievalParam
from federatedml.secureprotol.oblivious_transfer.hauck_oblivious_transfer.hauck_oblivious_transfer_sender import \
    HauckObliviousTransferSender
from federatedml.secureprotol.symmetric_encryption.py_aes_encryption import AESEncryptKey
from federatedml.util import consts, abnormal_detection, LOGGER


MODEL_PARAM_NAME = 'SecureInformationRetrievalParam'
MODEL_META_NAME = 'SecureInformationRetrievalMeta'


class SecureInformationRetrievalHost(BaseSecureInformationRetrieval):
    def __init__(self):
        super(SecureInformationRetrievalHost, self).__init__()
        self.oblivious_transfer = None

    def _init_model(self, param: SecureInformationRetrievalParam):
        self._init_base_model(param)

        if self.model_param.oblivious_transfer_protocol == consts.OT_HAUCK:
            self.oblivious_transfer = HauckObliviousTransferSender()
        else:
            LOGGER.error("SIR only supports Hauck's OT")

    def fit(self, data_inst):
        """

        :param data_inst: Table, only the key and the value (Instance.label) are used
        :return:
        """
        LOGGER.info("data count = {}".format(data_inst.count()))
        # 0. Raw retrieval
        if self.model_param.raw_retrieval or self.security_level == 0:
            LOGGER.info("enter raw information retrieval host")
            abnormal_detection.empty_table_detection(data_inst)
            self._raw_information_retrieval(data_inst)
            self._display_result(block_num='N/A')
            return data_inst

        # 1. Data pre-processing
        LOGGER.info("enter secure information retrieval host")
        abnormal_detection.empty_table_detection(data_inst)
        self._parse_security_level(data_inst)
        if not self._check_oblivious_transfer_condition():
            self._failure_response()

        # 2. Sync commutative cipher public knowledge, block num and init
        self._sync_commutative_cipher_public_knowledge()
        self.commutative_cipher.init()
        LOGGER.info("commutative cipher key generated")

        # 3. 1st ID encryption and exchange
        # g: guest's plaintext
        # Eg: guest's ciphertext
        # EEg: guest's doubly encrypted ciphertext
        # h, Eh, EEh: host
        # i, Ei, EEi: intersection
        id_list_host_first = self._encrypt_id(data_inst, reserve_value=True)      # (h, (Eh, Instance))
        LOGGER.info("encrypted host id for the 1st time")
        id_list_guest_first = self._exchange_id_list(
            id_list_host_first.map(lambda k, v: (v[0], -1)))       # send (Eh, -1), get (Eg, -1)

        # 4. 2nd ID encryption and send doubly encrypted ID list to guest
        id_list_guest_second = self._encrypt_id(id_list_guest_first)         # (EEg, -1)
        LOGGER.info("encrypted host id for the 2nd time")
        self._sync_doubly_encrypted_id_list(id_list_guest_second)       # send (EEg, -1)

        # 5. Wait for guest to find intersection and re-index the messages
        LOGGER.info("waiting for guest to find intersection and perform natural indexation")

        # 6. Get the re-indexed doubly encrypted ID from guest
        id_blocks = self._iteratively_get_id_blocks()

        # 7. Restore value for the intersection
        id_blocks = self._restore_value(id_list_host_first, id_blocks)      # List[(Ei, val)]
        LOGGER.info("interested values restored")

        # 8. Execute OT as sender
        LOGGER.info("enter oblivious transfer protocol as a sender")
        key_list = self.oblivious_transfer.key_derivation(self.block_num)
        LOGGER.info("oblivious transfer key derived")

        # 9. Encrypt and transmit
        self._non_committing_encrypt(id_blocks, key_list)       # List[(Ei, Eval)]
        LOGGER.info("non-committing encryption and transmission completed")

        # 10. Get doubly encrypted ID list from guest
        id_list_intersect_cipher_cipher = self._sync_intersect_cipher_cipher()      # get (EEright, -1)

        # 11. Decrypt and send to guest
        id_list_intersect_cipher = self._decrypt_id(
            id_list_intersect_cipher_cipher, reserve_value=True)    # (EEright, Eright)
        LOGGER.info("decryption completed")
        self._sync_intersect_cipher(id_list_intersect_cipher)

        # 12. Slack
        self._sync_coverage(data_inst)
        self._display_result()
        LOGGER.info("secure information retrieval finished")

        return data_inst

    def _sync_nonce_list(self, nonce, time):
        self.transfer_variable.nonce_list.remote(nonce,
                                                 suffix=(time,),
                                                 role=consts.GUEST,
                                                 idx=0)
        # federation.remote(obj=nonce,
        #                   name=self.transfer_variable.nonce_list.name,
        #                   tag=self.transfer_variable.generate_transferid(
        #                       self.transfer_variable.nonce_list, time),
        #                   role=consts.GUEST,
        #                   idx=0)
        LOGGER.info("sent {}-th nonce to guest".format(time))

    def _sync_intersect_cipher(self, id_list):
        self.transfer_variable.intersect_cipher.remote(id_list,
                                                       role=consts.GUEST,
                                                       idx=0)
        # federation.remote(obj=id_list,
        #                   name=self.transfer_variable.intersect_cipher.name,
        #                   tag=self.transfer_variable.generate_transferid(self.transfer_variable.intersect_cipher),
        #                   role=consts.GUEST,
        #                   idx=0)
        LOGGER.info("send intersect cipher to guest")

    def _decrypt_id(self, data_instance, reserve_value=False):
        """
        (e, De) if reserve_value, otherwise (De, -1)
        :param data_instance:
        :param reserve_value:
        :return:
        """
        if reserve_value:
            return self.commutative_cipher.map_decrypt(data_instance, mode=0)
        else:
            return self.commutative_cipher.map_decrypt(data_instance, mode=1)

    def _sync_intersect_cipher_cipher(self, id_list=None):
        id_list_intersect_cipher_cipher = self.transfer_variable.intersect_cipher_cipher.get(idx=0)
        # id_list_intersect_cipher_cipher = federation.get(
        #     name=self.transfer_variable.intersect_cipher_cipher.name,
        #     tag=self.transfer_variable.generate_transferid(self.transfer_variable.intersect_cipher_cipher),
        #     idx=0
        # )
        LOGGER.info("got intersect cipher cipher from guest")
        return id_list_intersect_cipher_cipher

    def _transmit_value_ciphertext(self, id_block, time):
        self.transfer_variable.id_blocks_ciphertext.remote(id_block,
                                                           suffix=(time,),
                                                           role=consts.GUEST,
                                                           idx=0)
        # federation.remote(obj=id_block,
        #                   name=self.transfer_variable.id_blocks_ciphertext.name,
        #                   tag=self.transfer_variable.generate_transferid(self.transfer_variable.id_blocks_ciphertext,
        #                                                                  time),
        #                   role=consts.GUEST,
        #                   idx=0)
        LOGGER.info("sent {}-th id block ciphertext to guest".format(time))

    def _non_committing_encrypt(self, id_blocks, key_list):
        """
        Use non-committing cipher to encrypt id blocks
        :param id_blocks: List[(Ei, val)]
        :param key_list: List[ObliviousTransferKey]
        :return:
        """
        for i in range(self.block_num):
            if self.model_param.non_committing_encryption == consts.AES:
                aes_key = CryptoExecutor(AESEncryptKey(key_list[i].key))
            else:
                raise ValueError("only supports AES cipher for non-committing encryption")
            self._transmit_value_ciphertext(aes_key.map_values_encrypt(id_blocks[i], mode=0), time=i)
            self._sync_nonce_list(aes_key.get_nonce(), time=i)

            block_confirm = self.transfer_variable.block_confirm.get(idx=0,
                                                                     suffix=(i,))
            if block_confirm:
                continue

    @staticmethod
    def _restore_value(id_list_host, id_blocks):
        """

        :param id_list_host: (h, (Eh, Instance))
        :param id_blocks: List[(Ei, -1)]
        :return:
        """
        id_list_host_parse = id_list_host.map(lambda k, v: (v[0], v[1].label))     # (Eh, val)
        id_value_blocks = []
        for i in range(len(id_blocks)):
            restored_table = id_list_host_parse.join(id_blocks[i], lambda v, u: v)
            id_value_blocks.append(restored_table)
        return id_value_blocks

    def _sync_natural_indexation(self, id_list=None, time=None):
        id_list_natural_indexation = self.transfer_variable.natural_indexation.get(idx=0,
                                                                                   suffix=(time,))
        # id_list_natural_indexation = federation.get(name=self.transfer_variable.natural_indexation.name,
        #                                             tag=self.transfer_variable.generate_transferid(
        #                                                 self.transfer_variable.natural_indexation, time),
        #                                             idx=0)
        LOGGER.info("got naturally indexed block {} from guest".format(time))
        return id_list_natural_indexation

    def _sync_doubly_encrypted_id_list(self, id_list):
        self.transfer_variable.doubly_encrypted_id_list.remote(id_list,
                                                               role=consts.GUEST,
                                                               idx=0)
        # federation.remote(obj=id_list,
        #                   name=self.transfer_variable.doubly_encrypted_id_list.name,
        #                   tag=self.transfer_variable.generate_transferid(
        #                       self.transfer_variable.doubly_encrypted_id_list),
        #                   role=consts.GUEST,
        #                   idx=0)
        LOGGER.info("sent doubly encrypted id list to guest")

    def _parse_security_level(self, data_instance):
        self._sync_block_num()

    def _exchange_id_list(self, id_list_host):
        self.transfer_variable.id_ciphertext_list_exchange_h2g.remote(id_list_host,
                                                                      role=consts.GUEST,
                                                                      idx=0)
        # federation.remote(obj=id_list_host,
        #                   name=self.transfer_variable.id_ciphertext_list_exchange_h2g.name,
        #                   tag=self.transfer_variable.generate_transferid(
        #                       self.transfer_variable.id_ciphertext_list_exchange_h2g),
        #                   role=consts.GUEST,
        #                   idx=0)
        LOGGER.info("sent id 1st ciphertext list to guest")
        id_list_guest = self.transfer_variable.id_ciphertext_list_exchange_g2h.get(idx=0)
        # id_list_guest = federation.get(name=self.transfer_variable.id_ciphertext_list_exchange_g2h.name,
        #                                tag=self.transfer_variable.generate_transferid(
        #                                    self.transfer_variable.id_ciphertext_list_exchange_g2h),
        #                                idx=0)
        LOGGER.info("got id 1st ciphertext list from guest")
        return id_list_guest

    def _encrypt_id(self, data_instance, reserve_value=False):
        """
        Encrypt the key (ID) column of the input Table
        :param data_instance: Table
                reserve_value: (k, (enc_k, v)) if reserve_value = True,
                    otherwise set all values to be minus one (enc_k, -1)
        :return:
        """
        if reserve_value:
            return self.commutative_cipher.map_encrypt(data_instance, mode=3)
        else:
            return self.commutative_cipher.map_encrypt(data_instance, mode=1)

    def _sync_commutative_cipher_public_knowledge(self):
        self.commutative_cipher = self.transfer_variable.commutative_cipher_public_knowledge.get(idx=0)
        # self.commutative_cipher = federation.get(name=self.transfer_variable.commutative_cipher_public_knowledge.name,
        #                                          tag=self.transfer_variable.generate_transferid(
        #                                              self.transfer_variable.commutative_cipher_public_knowledge),
        #                                          idx=0)
        LOGGER.info("got commutative cipher public knowledge from host {}".format(self.commutative_cipher))

    def _sync_block_num(self):
        self.block_num = self.transfer_variable.block_num.get(idx=0)
        # self.block_num = federation.get(name=self.transfer_variable.block_num.name,
        #                                 tag=self.transfer_variable.generate_transferid(
        #                                     self.transfer_variable.block_num),
        #                                 idx=0)
        LOGGER.info("got block num {} from guest".format(self.block_num))

    def _raw_information_retrieval(self, data_instance):
        id_list_guest = self.transfer_variable.raw_id_list.get(idx=0)
        # id_list_guest = federation.get(name=self.transfer_variable.raw_id_list.name,
        #                                tag=self.transfer_variable.generate_transferid(
        #                                    self.transfer_variable.raw_id_list),
        #                                idx=0)
        LOGGER.info("got raw id list from guest")

        id_intersect = data_instance.join(id_list_guest, lambda v, u: v)

        self.transfer_variable.raw_value_list.remote(id_intersect,
                                                     role=consts.GUEST,
                                                     idx=0)
        # federation.remote(obj=id_intersect,
        #                   name=self.transfer_variable.raw_value_list.name,
        #                   tag=self.transfer_variable.generate_transferid(
        #                       self.transfer_variable.raw_value_list),
        #                   role=consts.GUEST,
        #                   idx=0)
        LOGGER.info("sent raw value list to guest")

        self._sync_coverage(data_instance)

    def _sync_coverage(self, data_instance):
        self.coverage = self.transfer_variable.coverage.get(idx=0) / data_instance.count()
        # self.coverage = federation.get(name=self.transfer_variable.coverage.name,
        #                                tag=self.transfer_variable.generate_transferid(
        #                                    self.transfer_variable.coverage),
        #                                idx=0) / data_instance.count()
        LOGGER.info("got coverage {} from guest".format(self.coverage))

    def _iteratively_get_id_blocks(self):
        """

        :return: List[Table]
        """
        id_blocks = [None for _ in range(self.block_num)]
        for i in range(self.block_num):
            id_block = self._sync_natural_indexation(time=i)    # get List[(Ei, -1)]
            id_blocks[i] = id_block

        return id_blocks
