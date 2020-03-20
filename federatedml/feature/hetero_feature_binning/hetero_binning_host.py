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

from arch.api.utils import log_utils
# from federatedml.feature.binning.base_binning import IVAttributes
from federatedml.feature.hetero_feature_binning.base_feature_binning import BaseHeteroFeatureBinning
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class HeteroFeatureBinningHost(BaseHeteroFeatureBinning):
    def fit(self, data_instances):
        """
        Apply binning method for both data instances in local party as well as the other one. Afterwards, calculate
        the specific metric value for specific columns.
        """
        self._abnormal_detection(data_instances)

        # self._parse_cols(data_instances)
        self._setup_bin_inner_param(data_instances, self.model_param)

        # Calculates split points of datas in self party
        split_points = self.binning_obj.fit_split_points(data_instances)

        if not self.model_param.local_only:
            self._sync_init_bucket(data_instances, split_points)
            if self.model_param.method == consts.OPTIMAL:
                self.optimal_binning_sync()

        if self.transform_type != 'woe':
            data_instances = self.transform(data_instances)
        self.set_schema(data_instances)
        self.data_output = data_instances
        return data_instances

    def _sync_init_bucket(self, data_instances, split_points, need_shuffle=False):

        # self._make_iv_obj(split_points)  # Save split points

        data_bin_table = self.binning_obj.get_data_bin(data_instances, split_points)
        LOGGER.debug("data_bin_table: {}".format(data_bin_table))

        # encrypted_label_table_id = self.transfer_variable.generate_transferid(self.transfer_variable.encrypted_label)
        encrypted_label_table = self.transfer_variable.encrypted_label.get(idx=0)

        LOGGER.info("Get encrypted_label_table from guest")

        encrypted_bin_sum = self.__static_encrypted_bin_label(data_bin_table, encrypted_label_table,
                                                              self.bin_inner_param.bin_cols_map, split_points)
        # LOGGER.debug("encrypted_bin_sum: {}".format(encrypted_bin_sum))

        if need_shuffle:
            encrypted_bin_sum = self.binning_obj.shuffle_static_counts(encrypted_bin_sum)

        encrypted_bin_sum = self.bin_inner_param.encode_col_name_dict(encrypted_bin_sum)
        self.transfer_variable.encrypted_bin_sum.remote(encrypted_bin_sum,
                                                        role=consts.GUEST,
                                                        idx=0)

    def __static_encrypted_bin_label(self, data_bin_table, encrypted_label, cols_dict, split_points):
        data_bin_with_label = data_bin_table.join(encrypted_label, lambda x, y: (x, y))
        f = functools.partial(self.binning_obj.add_label_in_partition,
                              split_points=split_points,
                              cols_dict=cols_dict)
        result_sum = data_bin_with_label.mapPartitions(f)
        encrypted_bin_sum = result_sum.reduce(self.binning_obj.aggregate_partition_label)
        return encrypted_bin_sum

    def optimal_binning_sync(self):
        bucket_idx = self.transfer_variable.bucket_idx.get(idx=0)
        LOGGER.debug("In optimal_binning_sync, received bucket_idx: {}".format(bucket_idx))
        original_split_points = self.binning_obj.bin_results.all_split_points
        for encoded_col_name, b_idx in bucket_idx.items():
            col_name = self.bin_inner_param.decode_col_name(encoded_col_name)
            ori_sp_list = original_split_points.get(col_name)
            optimal_result = [ori_sp_list[i] for i in b_idx]
            self.binning_obj.bin_results.put_col_split_points(col_name, optimal_result)
