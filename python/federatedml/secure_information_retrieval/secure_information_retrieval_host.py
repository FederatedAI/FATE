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
    BaseSecureInformationRetrieval
from federatedml.param.sir_param import SecureInformationRetrievalParam
from federatedml.param.intersect_param import IntersectParam
from federatedml.secureprotol.oblivious_transfer.hauck_oblivious_transfer.hauck_oblivious_transfer_sender import \
    HauckObliviousTransferSender
from federatedml.secureprotol.symmetric_encryption.py_aes_encryption import AESEncryptKey
from federatedml.secureprotol.symmetric_encryption.cryptor_executor import CryptoExecutor
from federatedml.statistic import data_overview
from federatedml.statistic.intersect import DhIntersectionHost
from federatedml.util import consts, abnormal_detection, LOGGER


MODEL_PARAM_NAME = 'SecureInformationRetrievalParam'
MODEL_META_NAME = 'SecureInformationRetrievalMeta'


class SecureInformationRetrievalHost(BaseSecureInformationRetrieval):
    def __init__(self):
        super(SecureInformationRetrievalHost, self).__init__()
        self.oblivious_transfer = None
        self.target_indexes = None

    def _init_model(self, param: SecureInformationRetrievalParam):
        self._init_base_model(param)

        self.intersection_obj = DhIntersectionHost()
        self.intersection_obj.role = consts.HOST
        intersect_param = IntersectParam(dh_params=self.dh_params)
        self.intersection_obj.load_params(intersect_param)
        self.intersection_obj.host_party_id_list = self.component_properties.host_party_idlist
        self.intersection_obj.guest_party_id = self.component_properties.guest_partyid

        if self.model_param.oblivious_transfer_protocol == consts.OT_HAUCK.lower():
            self.oblivious_transfer = HauckObliviousTransferSender()
        else:
            raise ValueError("SIR only supports Hauck's OT")

    def fit(self, data_inst):
        """

        :param data_inst: Table
        :return:
        """
        # LOGGER.info("data count = {}".format(data_inst.count()))
        abnormal_detection.empty_table_detection(data_inst)
        self._update_target_indexes(data_inst.schema)
        match_data = data_inst
        if data_overview.check_with_inst_id(data_inst):
            match_data = self._recover_match_id(data_inst)

        # 0. Raw retrieval
        if self.model_param.raw_retrieval or self.security_level == 0:
            LOGGER.info("enter raw information retrieval host")
            # abnormal_detection.empty_table_detection(data_inst)
            self._raw_information_retrieval(match_data)
            self._display_result(block_num='N/A')
            self._sync_coverage(data_inst)
            return data_inst

        # 1. Data pre-processing
        LOGGER.info("enter secure information retrieval host")
        # abnormal_detection.empty_table_detection(data_inst)
        self._parse_security_level(match_data)
        if not self._check_oblivious_transfer_condition():
            self._failure_response()

        # 2. Guest find intersection
        self.intersection_obj.get_intersect_doubly_encrypted_id(match_data)
        id_list_host_first = self.intersection_obj.id_list_local_first

        # 3. Get the re-indexed doubly encrypted ID from guest
        id_blocks = self._iteratively_get_id_blocks()

        # 4. Restore value for the intersection
        id_blocks = _restore_value(id_list_host_first,
                                   id_blocks,
                                   self.target_indexes,
                                   self.need_label)      # List[(Ei, val)]
        LOGGER.info("interested values restored")

        # 8. Execute OT as sender
        LOGGER.info("enter oblivious transfer protocol as a sender")
        key_list = self.oblivious_transfer.key_derivation(self.block_num)
        LOGGER.info("oblivious transfer key derived")

        # 9. Encrypt and transmit
        self._non_committing_encrypt(id_blocks, key_list)       # List[(Ei, Eval)]
        LOGGER.info("non-committing encryption and transmission completed")

        # 10. Slack
        self._sync_coverage(data_inst)
        self._display_result()
        LOGGER.info("secure information retrieval finished")

        return data_inst

    def _sync_nonce_list(self, nonce, time):
        self.transfer_variable.nonce_list.remote(nonce,
                                                 suffix=(time,),
                                                 role=consts.GUEST,
                                                 idx=0)
        LOGGER.info("sent {}-th nonce to guest".format(time))

    def _transmit_value_ciphertext(self, id_block, time):
        self.transfer_variable.id_blocks_ciphertext.remote(id_block,
                                                           suffix=(time,),
                                                           role=consts.GUEST,
                                                           idx=0)
        LOGGER.info("sent {}-th id block ciphertext to guest".format(time))

    def _non_committing_encrypt(self, id_blocks, key_list):
        """
        Use non-committing cipher to encrypt id blocks
        :param id_blocks: List[(Ei, val)]
        :param key_list: List[ObliviousTransferKey]
        :return:
        """
        for i in range(self.block_num):
            if self.model_param.non_committing_encryption == consts.AES.lower():
                aes_key = CryptoExecutor(AESEncryptKey(key_list[i].key))
            else:
                raise ValueError("only supports AES cipher for non-committing encryption")
            self._transmit_value_ciphertext(aes_key.map_values_encrypt(id_blocks[i], mode=0), time=i)
            self._sync_nonce_list(aes_key.get_nonce(), time=i)

            block_confirm = self.transfer_variable.block_confirm.get(idx=0,
                                                                     suffix=(i,))
            if block_confirm:
                continue

    def _update_target_indexes(self, schema):
        self.need_label = self._check_need_label()
        if self.need_label:
            return
        header = schema["header"]
        target_indexes = []
        for col_name in self.target_cols:
            try:
                i = header.index(col_name)
                target_indexes.append(i)
            except ValueError:
                raise ValueError(f"{col_name} does not exist in table header. Please check.")
        self.target_indexes = target_indexes

    @staticmethod
    def extract_value(instance, target_indexes, need_label):
        if need_label:
            return instance.label
        features = [instance.features[i] for i in target_indexes]
        return features

    def _sync_natural_indexation(self, id_list=None, time=None):
        id_list_natural_indexation = self.transfer_variable.natural_indexation.get(idx=0,
                                                                                   suffix=(time,))
        LOGGER.info(f"got naturally indexed block {time} from guest")
        return id_list_natural_indexation

    def _parse_security_level(self, data_instance):
        self._sync_block_num()

    def _sync_block_num(self):
        self.block_num = self.transfer_variable.block_num.get(idx=0)
        LOGGER.info("got block num {} from guest".format(self.block_num))

    def _raw_information_retrieval(self, data_instance):
        id_list_guest = self.transfer_variable.raw_id_list.get(idx=0)
        LOGGER.info("got raw id list from guest")
        target_indexes, need_label = self.target_indexes, self.need_label

        id_intersect = data_instance.join(id_list_guest,
                                          lambda v, u: SecureInformationRetrievalHost.extract_value(v,
                                                                                                    target_indexes,
                                                                                                    need_label))

        self.transfer_variable.raw_value_list.remote(id_intersect,
                                                     role=consts.GUEST,
                                                     idx=0)
        LOGGER.info("sent raw value list to guest")

        # self._sync_coverage(data_instance)

    def _sync_coverage(self, data_instance):
        self.coverage = self.transfer_variable.coverage.get(idx=0) / data_instance.count()
        LOGGER.info(f"got coverage {self.coverage} from guest")

    def _iteratively_get_id_blocks(self):
        """

        :return: List[Table]
        """
        id_blocks = [None for _ in range(self.block_num)]
        for i in range(self.block_num):
            id_block = self._sync_natural_indexation(time=i)    # get List[(Ei, -1)]
            id_blocks[i] = id_block

        return id_blocks


def _restore_value(id_list_host, id_blocks, target_indexes, need_label):
    """

    :param id_list_host: (h, (Eh, Instance))
    :param id_blocks: List[(Ei, -1)]
    :return:
    """
    id_value_blocks = []
    for i in range(len(id_blocks)):
        restored_table = id_list_host.join(id_blocks[i],
                                           lambda v, u:
                                           SecureInformationRetrievalHost.extract_value(v[1],
                                                                                        target_indexes,
                                                                                        need_label))
        id_value_blocks.append(restored_table)
    return id_value_blocks
