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

import functools

from arch.api import federation
from arch.api.utils import log_utils
from federatedml.feature.hetero_feature_binning.base_feature_binning import BaseHeteroFeatureBinning
from federatedml.param.param import FeatureBinningParam
from federatedml.secureprotol import PaillierEncrypt
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class HeteroFeatureBinningGuest(BaseHeteroFeatureBinning):
    def __init__(self, params: FeatureBinningParam):
        super(HeteroFeatureBinningGuest, self).__init__(params)

        self.encryptor = PaillierEncrypt()
        self.encryptor.generate_key()
        self.iv_attrs = None
        self.host_iv_attrs = None

    def fit(self, data_instances):
        """
        Apply binning method for both data instances in local party as well as the other one. Afterwards, calculate
        the specific metric value for specific columns.
        """
        self._abnormal_detection(data_instances)
        self._parse_cols(data_instances)

        # 1. Synchronize encryption information
        self.__synchronize_encryption()

        # 2. Prepare labels
        data_instances = data_instances.mapValues(self.load_data)
        label_table = data_instances.mapValues(lambda x: x.label)

        # 3. Transfer encrypted label
        f = functools.partial(self.encrypt,
                              encryptor=self.encryptor)
        encrypted_label_table = label_table.mapValues(f)

        encrypted_label_table_id = self.transfer_variable.generate_transferid(self.transfer_variable.encrypted_label)
        federation.remote(encrypted_label_table, name=self.transfer_variable.encrypted_label.name,
                          tag=encrypted_label_table_id, role=consts.HOST, idx=0)

        LOGGER.info("Sent encrypted_label_table to host")

        # 4. Calculates self's binning. In case the other party need time to compute its data,
        #  do binning calculation at this point.
        local_iv = self.fit_local(data_instances, label_table)

        # 5. Received host result and calculate iv value
        encrypted_bin_sum_id = self.transfer_variable.generate_transferid(self.transfer_variable.encrypted_bin_sum)

        encrypted_bin_sum = federation.get(name=self.transfer_variable.encrypted_bin_sum.name,
                                           tag=encrypted_bin_sum_id,
                                           idx=0)

        LOGGER.info("Get encrypted_bin_sum from host")

        result_counts = self.__decrypt_bin_sum(encrypted_bin_sum)
        host_iv_attrs = self.binning_obj.cal_iv_woe(result_counts, self.bin_param.adjustment_factor)
        self.host_iv_attrs = host_iv_attrs
        # LOGGER.debug("Lenght of host iv attrs: {}".format(len(self.host_iv_attrs)))
        # for idx, col in enumerate(self.cols):
        #     LOGGER.info("The local iv of {}th feature is {}".format(col, local_iv[idx].iv))

        for idx, iv_attr in enumerate(host_iv_attrs):
            LOGGER.info("The remote iv of {}th measured feature is {}".format(idx, iv_attr.iv))

        iv_result = {'local': local_iv,
                     'remote': host_iv_attrs}

        return iv_result

    def transform(self, data_instances):
        self._abnormal_detection(data_instances)

        self.header = data_instances.schema.get('header')  # ['x1', 'x2', 'x3' ... ]

        self._parse_cols(data_instances)

        # 1. Synchronize encryption information
        self.__synchronize_encryption()

        # 2. Prepare labels
        data_instances = data_instances.mapValues(self.load_data)
        label_table = data_instances.mapValues(lambda x: x.label)

        # 3. Transfer encrypted label
        f = functools.partial(self.encrypt,
                              encryptor=self.encryptor)
        encrypted_label_table = label_table.mapValues(f)
        encrypted_label_table_id = self.transfer_variable.generate_transferid(self.transfer_variable.encrypted_label)
        federation.remote(encrypted_label_table, name=self.transfer_variable.encrypted_label.name,
                          tag=encrypted_label_table_id, role=consts.HOST, idx=0)
        LOGGER.info("Sent encrypted_label_table to host for transform")

        # 4. Transform locally
        self.transform_local(data_instances, reformated=True)

        # 5. Received host result and calculate iv value
        encrypted_bin_sum_id = self.transfer_variable.generate_transferid(self.transfer_variable.encrypted_bin_sum)
        encrypted_bin_sum = federation.get(name=self.transfer_variable.encrypted_bin_sum.name,
                                           tag=encrypted_bin_sum_id,
                                           idx=0)

        result_counts = self.__decrypt_bin_sum(encrypted_bin_sum)
        host_iv_attrs = self.binning_obj.cal_iv_woe(result_counts, self.bin_param.adjustment_factor)
        self.host_iv_attrs = host_iv_attrs
        for idx, iv_attr in enumerate(host_iv_attrs):
            LOGGER.info("The remote iv of {}th measured feature is {}".format(idx, iv_attr.iv))

        data_instances.schema['header'] = self.header
        return data_instances

    @staticmethod
    def encrypt(x, encryptor):
        return encryptor.encrypt(x), encryptor.encrypt(1 - x)

    def transform_local(self, data_instances, reformated=False):
        self._abnormal_detection(data_instances)

        self._parse_cols(data_instances)

        if not reformated:  # Reformat the label type
            data_instances = data_instances.mapValues(self.load_data)

        split_points = []
        for iv_attr in self.iv_attrs:
            s_p = list(iv_attr.split_points)
            split_points.append(s_p)

        self.iv_attrs = self.binning_obj.cal_local_iv(data_instances, self.cols, split_points)
        for idx, col in enumerate(self.cols):
            LOGGER.info("The local iv of {}th feature is {}".format(col, self.iv_attrs[idx].iv))

    def __synchronize_encryption(self):
        pub_key = self.encryptor.get_public_key()
        pubkey_id = self.transfer_variable.generate_transferid(self.transfer_variable.paillier_pubkey)
        # LOGGER.debug("pubkey_id is : {}".format(pubkey_id))

        federation.remote(pub_key, name=self.transfer_variable.paillier_pubkey.name,
                          tag=pubkey_id, role=consts.HOST, idx=0)

        LOGGER.info("send pubkey to host")
        self.has_synchronized = True

    def __decrypt_bin_sum(self, encrypted_bin_sum):
        for feature_sum in encrypted_bin_sum:
            for idx, (encrypted_event, encrypted_non_event) in enumerate(feature_sum):
                event_count = self.encryptor.decrypt(encrypted_event)
                non_event_count = self.encryptor.decrypt(encrypted_non_event)
                feature_sum[idx] = (event_count, non_event_count)
        return encrypted_bin_sum

    def fit_local(self, data_instances, label_table=None):
        self._abnormal_detection(data_instances)

        self._parse_cols(data_instances)

        iv_attrs = self.binning_obj.cal_local_iv(data_instances, self.cols, label_table=label_table)
        for idx, col in enumerate(self.cols):
            LOGGER.info("The local iv of {}th feature is {}".format(col, iv_attrs[idx].iv))
        self.iv_attrs = iv_attrs
        return iv_attrs

    @staticmethod
    def load_data(data_instance):
        # Here suppose this is a binary question and the event label is 1
        # LOGGER.debug('label type is {}'.format(type(data_instance.label)))
        if data_instance.label != 1:
            data_instance.label = 0
        return data_instance

