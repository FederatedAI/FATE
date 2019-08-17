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
        self.local_transform_result = None
        self.party_name = consts.GUEST
        self._init_binning_obj()

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
        self.set_schema(data_instances)

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
        data_instances = self.fit_local(data_instances, label_table)

        # 5. Received host result and calculate iv value
        encrypted_bin_sum_id = self.transfer_variable.generate_transferid(self.transfer_variable.encrypted_bin_sum)

        encrypted_bin_sum = federation.get(name=self.transfer_variable.encrypted_bin_sum.name,
                                           tag=encrypted_bin_sum_id,
                                           idx=0)

        LOGGER.info("Get encrypted_bin_sum from host")

        result_counts = self.__decrypt_bin_sum(encrypted_bin_sum)
        host_iv_attrs = self.binning_obj.cal_iv_woe(result_counts, self.bin_param.adjustment_factor)

        # Support one host only in this version. Multiple host will be supported in the future.
        self.host_results[consts.HOST] = host_iv_attrs

        for cols_name, iv_attr in host_iv_attrs.items():
            display_result = iv_attr.display_result(self.bin_param.display_result)
            LOGGER.info("[Result][FeatureBinning][Host] feature {} 's result is : {}".format(cols_name, display_result))

        self.set_schema(data_instances)
        return data_instances

    def transform(self, data_instances):
        self._abnormal_detection(data_instances)
        self._parse_cols(data_instances)

        # 1. Synchronize encryption information
        self.__synchronize_encryption()

        # 2. Prepare labels
        data_instances = data_instances.mapValues(self.load_data)
        label_table = data_instances.mapValues(lambda x: x.label)
        self.set_schema(data_instances)

        # 3. Transfer encrypted label
        f = functools.partial(self.encrypt,
                              encryptor=self.encryptor)
        encrypted_label_table = label_table.mapValues(f)
        encrypted_label_table_id = self.transfer_variable.generate_transferid(self.transfer_variable.encrypted_label)
        federation.remote(encrypted_label_table, name=self.transfer_variable.encrypted_label.name,
                          tag=encrypted_label_table_id, role=consts.HOST, idx=0)
        LOGGER.info("Sent encrypted_label_table to host for transform")

        # 4. Transform locally
        self.transform_local(data_instances, label_table=label_table, save_result=False)

        # 5. Received host result and calculate iv value
        encrypted_bin_sum_id = self.transfer_variable.generate_transferid(self.transfer_variable.encrypted_bin_sum)
        encrypted_bin_sum = federation.get(name=self.transfer_variable.encrypted_bin_sum.name,
                                           tag=encrypted_bin_sum_id,
                                           idx=0)

        result_counts = self.__decrypt_bin_sum(encrypted_bin_sum)
        host_iv_attrs = self.binning_obj.cal_iv_woe(result_counts, self.bin_param.adjustment_factor)
        # host_results = {'host1': host_iv_attrs}

        # self.save_model(name=self.bin_param.transform_table,
        #                 namespace=self.bin_param.result_namespace,
        #                 binning_result=self.local_transform_result,
        #                 host_results=host_results)

        for col_name, iv_attr in host_iv_attrs.items():
            LOGGER.info("The remote feature {} 's iv is {}".format(col_name, iv_attr.iv))

        self.set_schema(data_instances)
        return data_instances

    @staticmethod
    def encrypt(x, encryptor):
        return encryptor.encrypt(x), encryptor.encrypt(1 - x)

    def transform_local(self, data_instances, label_table=None, save_result=True):
        self._abnormal_detection(data_instances)
        self._parse_cols(data_instances)
        split_points = {}
        for col_name, iv_attr in self.binning_result.items():
            split_points[col_name] = iv_attr.split_points

        self.local_transform_result = self.binning_obj.cal_local_iv(data_instances,
                                                                    split_points=split_points,
                                                                    label_table=label_table)

        if save_result:
            self.save_model(name=self.bin_param.transform_table,
                            namespace=self.bin_param.result_namespace,
                            binning_result=self.local_transform_result,
                            host_results={})
        for col_name, col_index in self.local_transform_result.items():
            LOGGER.info("The local feature {} 's iv is {}".format(col_name, self.local_transform_result[col_name].iv))
        self.set_schema(data_instances)
        return data_instances

    def __synchronize_encryption(self):
        pub_key = self.encryptor.get_public_key()
        pubkey_id = self.transfer_variable.generate_transferid(self.transfer_variable.paillier_pubkey)

        federation.remote(pub_key, name=self.transfer_variable.paillier_pubkey.name,
                          tag=pubkey_id, role=consts.HOST, idx=0)

        LOGGER.info("send pubkey to host")
        self.has_synchronized = True

    def __decrypt_bin_sum(self, encrypted_bin_sum):
        # for feature_sum in encrypted_bin_sum:
        for col_name, count_list in encrypted_bin_sum.items():
            new_list = []
            for encrypted_event, encrypted_non_event in count_list:
                event_count = self.encryptor.decrypt(encrypted_event)
                non_event_count = self.encryptor.decrypt(encrypted_non_event)
                new_list.append((event_count, non_event_count))
            encrypted_bin_sum[col_name] = new_list
        return encrypted_bin_sum

    def fit_local(self, data_instances, label_table=None):
        self._abnormal_detection(data_instances)
        self._parse_cols(data_instances)

        iv_attrs = self.binning_obj.cal_local_iv(data_instances, label_table=label_table)
        for col_name, iv_attr in iv_attrs.items():
            display_result = iv_attr.display_result(self.bin_param.display_result)
            LOGGER.info("[Result][FeatureBinning][Guest] feature {} 's result is : {}".format(col_name, display_result))
            # LOGGER.info("[Result][FeatureBinning]The feature {} 's iv is {}".format(col_name, iv_attrs[col_name].iv))
        self.binning_result = iv_attrs
        self.set_schema(data_instances)
        return data_instances

    @staticmethod
    def load_data(data_instance):
        # Here suppose this is a binary question and the event label is 1
        if data_instance.label != 1:
            data_instance.label = 0
        return data_instance
