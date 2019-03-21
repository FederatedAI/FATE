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
from federatedml.feature.binning import QuantileBinning
from federatedml.param.param import FeatureBinningParam
from federatedml.secureprotol import PaillierEncrypt
from federatedml.util import consts
from federatedml.util.fate_operator import get_features_shape
from federatedml.util.transfer_variable import HeteroFeatureBinningTransferVariable

LOGGER = log_utils.getLogger()


class HeteroFeatureGuest(object):
    def __init__(self, params: FeatureBinningParam):
        self.bin_param = params
        if self.bin_param.method == consts.QUANTILE:
            self.binning_obj = QuantileBinning(self.bin_param)
        else:
            # LOGGER.warning("bin method: {} is not support yet. Change to quantile binning".format(
            #     self.bin_param.method
            # ))
            self.binning_obj = QuantileBinning(self.bin_param)
        self.cols = params.cols

        self.encryptor = PaillierEncrypt()
        self.encryptor.generate_key()
        self.transfer_variable = HeteroFeatureBinningTransferVariable()

    def fit(self, data_instances):
        """
        Apply binning method for both data instances in local party as well as the other one. Afterwards, calculate
        the specific metric value for specific columns.
        """
        if self.cols == -1:
            features_shape = get_features_shape(data_instances)
            if features_shape is None:
                raise RuntimeError('Cannot get feature shape, please check input data')
            self.cols = [i for i in range(features_shape)]

        # 1. Synchronize encryption information
        self.__synchronize_encryption()

        # 2. Prepare labels
        data_instances = data_instances.mapValues(self.load_data)
        label_table = data_instances.mapValues(lambda x: x.label)

        # 3. Transfer encrypted label
        encrypted_label_table = label_table.mapValues(lambda x: (self.encryptor.encrypt(x),
                                                                 self.encryptor.encrypt(1 - x)))
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
        iv_attrs = self.binning_obj.cal_iv_woe(result_counts, self.bin_param.adjustment_factor)

        # for idx, col in enumerate(self.cols):
        #     LOGGER.info("The local iv of {}th feature is {}".format(col, local_iv[idx].iv))

        for idx, iv_attr in enumerate(iv_attrs):
            LOGGER.info("The remote iv of {}th measured feature is {}".format(idx, iv_attr.iv))

        iv_result = {'local': local_iv,
                     'remote': iv_attrs}
        return iv_result

    def __synchronize_encryption(self):
        pub_key = self.encryptor.get_public_key()
        pubkey_id = self.transfer_variable.generate_transferid(self.transfer_variable.paillier_pubkey)
        federation.remote(pub_key, name=self.transfer_variable.paillier_pubkey.name,
                          tag=pubkey_id, role=consts.HOST, idx=0)
        LOGGER.debug("send pubkey to host")

    def __decrypt_bin_sum(self, encrypted_bin_sum):
        for feature_sum in encrypted_bin_sum:
            for idx, (encrypted_event, encrypted_non_event) in enumerate(feature_sum):
                event_count = self.encryptor.decrypt(encrypted_event)
                non_event_count = self.encryptor.decrypt(encrypted_non_event)
                feature_sum[idx] = (event_count, non_event_count)
        return encrypted_bin_sum

    def fit_local(self, data_instances, label_table=None):
        if self.cols == -1:
            features_shape = get_features_shape(data_instances)
            if features_shape is None:
                raise RuntimeError('Cannot get feature shape, please check input data')
            self.cols = [i for i in range(features_shape)]

        split_points = self.binning_obj.binning(data_instances, cols=self.cols)
        data_bin_table = self.binning_obj.transform(data_instances, split_points, self.cols)
        if label_table is None:
            label_table = data_instances.mapValues(lambda x: x.label)
        event_count_table = label_table.mapValues(lambda x: (x, 1 - x))
        data_bin_with_label = data_bin_table.join(event_count_table, lambda x, y: (x, y))
        f = functools.partial(self.binning_obj.add_label_in_partition,
                              total_bin=self.bin_param.bin_num,
                              cols=self.cols)
        result_sum = data_bin_with_label.mapPartitions(f)
        result_counts = result_sum.reduce(self.binning_obj.aggregate_partition_label)
        iv_attrs = self.binning_obj.cal_iv_woe(result_counts, self.bin_param.adjustment_factor,
                                               split_points=split_points)
        for idx, col in enumerate(self.cols):
            LOGGER.info("The local iv of {}th feature is {}".format(col, iv_attrs[idx].iv))

        return iv_attrs

    @staticmethod
    def load_data(data_instance):
        # Here suppose this is a binary question and the event label is 1
        # LOGGER.debug('label type is {}'.format(type(data_instance.label)))
        if data_instance.label != 1:
            data_instance.label = 0
        return data_instance
