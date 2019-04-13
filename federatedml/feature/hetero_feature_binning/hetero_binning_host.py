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
from federatedml.feature.binning import IVAttributes
from federatedml.feature.hetero_feature_binning.base_feature_binning import BaseHeteroFeatureBinning
from federatedml.param.param import FeatureBinningParam
from federatedml.secureprotol import PaillierEncrypt
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class HeteroFeatureBinningHost(BaseHeteroFeatureBinning):
    def __init__(self, params: FeatureBinningParam):
        super(HeteroFeatureBinningHost, self).__init__(params)

        self.encryptor = PaillierEncrypt()
        self.iv_attrs = None

    def fit(self, data_instances):
        """
        Apply binning method for both data instances in local party as well as the other one. Afterwards, calculate
        the specific metric value for specific columns.
        """
        self._abnormal_detection(data_instances)

        self._parse_cols(data_instances)

        # 1. Synchronize encryption information
        self.__synchronize_encryption()

        # Calculates split points of datas in self party
        split_points = self.binning_obj.binning(data_instances, cols=self.cols)
        self._make_iv_obj(split_points)
        # LOGGER.debug("host split_points are: {}".format(split_points))
        LOGGER.debug("Before transform, self cols: {}".format(self.cols))
        data_bin_table = self.binning_obj.transform(data_instances, split_points, self.cols)

        encrypted_label_table_id = self.transfer_variable.generate_transferid(self.transfer_variable.encrypted_label)
        encrypted_label_table = federation.get(name=self.transfer_variable.encrypted_label.name,
                                               tag=encrypted_label_table_id,
                                               idx=0)

        LOGGER.info("Get encrypted_label_table from guest")

        encrypted_bin_sum = self.__static_encrypted_bin_label(data_bin_table, encrypted_label_table, self.cols)
        encrypted_bin_sum_id = self.transfer_variable.generate_transferid(self.transfer_variable.encrypted_bin_sum)

        federation.remote(encrypted_bin_sum,
                          name=self.transfer_variable.encrypted_bin_sum.name,
                          tag=encrypted_bin_sum_id,
                          role=consts.GUEST,
                          idx=0)

        LOGGER.info("Sent encrypted_bin_sum to guest")

    def transform(self, data_instances):
        self._abnormal_detection(data_instances)

        self._parse_cols(data_instances)

        # 1. Synchronize encryption information
        self.__synchronize_encryption()

        split_points = []
        for iv_attr in self.iv_attrs:
            s_p = list(iv_attr.split_points)
            split_points.append(s_p)

        # LOGGER.debug("In transform, self.cols: {}".format(self.cols))
        data_bin_table = self.binning_obj.transform(data_instances, split_points, self.cols)

        encrypted_label_table_id = self.transfer_variable.generate_transferid(self.transfer_variable.encrypted_label)
        encrypted_label_table = federation.get(name=self.transfer_variable.encrypted_label.name,
                                               tag=encrypted_label_table_id,
                                               idx=0)
        LOGGER.info("Get encrypted_label_table from guest")

        encrypted_bin_sum = self.__static_encrypted_bin_label(data_bin_table, encrypted_label_table, self.cols)
        encrypted_bin_sum_id = self.transfer_variable.generate_transferid(self.transfer_variable.encrypted_bin_sum)
        federation.remote(encrypted_bin_sum,
                          name=self.transfer_variable.encrypted_bin_sum.name,
                          tag=encrypted_bin_sum_id,
                          role=consts.GUEST,
                          idx=0)
        LOGGER.info("Sent encrypted_bin_sum to guest")

    def _make_iv_obj(self, split_points):
        iv_objs = []
        for s_p in split_points:
            iv_obj = IVAttributes([], [], [], [], [], [], s_p)
            iv_objs.append(iv_obj)
        self.iv_attrs = iv_objs

    def __synchronize_encryption(self):
        pubkey_id = self.transfer_variable.generate_transferid(self.transfer_variable.paillier_pubkey)
        pubkey = federation.get(name=self.transfer_variable.paillier_pubkey.name,
                                tag=pubkey_id,
                                idx=0)

        LOGGER.info("Received pub_key from guest")
        self.encryptor.set_public_key(pubkey)
        self.has_synchronized = True

    def __static_encrypted_bin_label(self, data_bin_table, encrypted_label, cols):
        data_bin_with_label = data_bin_table.join(encrypted_label, lambda x, y: (x, y))
        f = functools.partial(self.binning_obj.add_label_in_partition,
                              total_bin=self.bin_param.bin_num,
                              cols=cols,
                              encryptor=self.encryptor)
        result_sum = data_bin_with_label.mapPartitions(f)
        encrypted_bin_sum = result_sum.reduce(self.binning_obj.aggregate_partition_label)
        return encrypted_bin_sum


