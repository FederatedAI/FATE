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
import operator

from federatedml.cipher_compressor import compressor
from federatedml.feature.hetero_feature_binning.base_feature_binning import BaseFeatureBinning
from federatedml.secureprotol.fate_paillier import PaillierEncryptedNumber
from federatedml.util import LOGGER
from federatedml.util import consts


class HeteroFeatureBinningHost(BaseFeatureBinning):
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
        if self.model_param.skip_static:
            if self.transform_type != 'woe':
                data_instances = self.transform(data_instances)
            self.set_schema(data_instances)
            self.data_output = data_instances
            return data_instances

        if not self.model_param.local_only:
            self._sync_init_bucket(data_instances, split_points)
            if self.model_param.method == consts.OPTIMAL:
                self.optimal_binning_sync()

        if self.transform_type != 'woe':
            data_instances = self.transform(data_instances)
        self.set_schema(data_instances)
        self.data_output = data_instances
        total_summary = self.binning_obj.bin_results.summary()
        self.set_summary(total_summary)
        return data_instances

    def _sync_init_bucket(self, data_instances, split_points, need_shuffle=False):

        data_bin_table = self.binning_obj.get_data_bin(data_instances, split_points)
        LOGGER.debug("data_bin_table, count: {}".format(data_bin_table.count()))

        encrypted_label_table = self.transfer_variable.encrypted_label.get(idx=0)

        LOGGER.info("Get encrypted_label_table from guest")

        encrypted_bin_sum = self.__static_encrypted_bin_label(data_bin_table, encrypted_label_table)

        encode_name_f = functools.partial(self.bin_inner_param.encode_col_name_dict,
                                          model=self,
                                          col_name_maps=self.bin_inner_param.col_name_maps)
        # encrypted_bin_sum = self.bin_inner_param.encode_col_name_dict(encrypted_bin_sum, self)
        encrypted_bin_sum = encrypted_bin_sum.map(encode_name_f)

        self.header_anonymous = self.bin_inner_param.encode_col_name_list(self.header, self)
        encrypted_bin_sum = self.cipher_compress(encrypted_bin_sum, data_bin_table.count())
        self.transfer_variable.encrypted_bin_sum.remote(encrypted_bin_sum,
                                                        role=consts.GUEST,
                                                        idx=0)
        send_result = {
            "category_names": self.bin_inner_param.encode_col_name_list(
                self.bin_inner_param.category_names, self),
            "bin_method": self.model_param.method,
            "optimal_params": {
                "metric_method": self.model_param.optimal_binning_param.metric_method,
                "bin_num": self.model_param.bin_num,
                "mixture": self.model_param.optimal_binning_param.mixture,
                "max_bin_pct": self.model_param.optimal_binning_param.max_bin_pct,
                "min_bin_pct": self.model_param.optimal_binning_param.min_bin_pct
            }
        }
        self.transfer_variable.optimal_info.remote(send_result,
                                                        role=consts.GUEST,
                                                        idx=0)

    def __static_encrypted_bin_label(self, data_bin_table, encrypted_label):
        data_bin_with_label = data_bin_table.join(encrypted_label, lambda x, y: (x, y))
        label_counts = encrypted_label.reduce(operator.add)
        sparse_bin_points = self.binning_obj.get_sparse_bin(self.bin_inner_param.bin_indexes,
                                                            self.binning_obj.split_points)
        sparse_bin_points = {self.bin_inner_param.header[k]: v for k, v in sparse_bin_points.items()}

        f = functools.partial(self.binning_obj.add_label_in_partition,
                              sparse_bin_points=sparse_bin_points)
        encrypted_bin_sum = data_bin_with_label.mapReducePartitions(f, self.binning_obj.aggregate_partition_label)
        f = functools.partial(self.binning_obj.fill_sparse_result,
                              sparse_bin_points=sparse_bin_points,
                              label_counts=label_counts)
        encrypted_bin_sum = encrypted_bin_sum.map(f)

        return encrypted_bin_sum

    def __static_encrypted_bin_label_deprecated(self, data_bin_table, encrypted_label, cols_dict, split_points):
        """
        Returns:
            table with value like:
                [[event_count, total_num], [event_count, total_num] ... ]
        """
        data_bin_with_label = data_bin_table.join(encrypted_label, lambda x, y: (x, y))
        event_sum = encrypted_label.reduce(operator.add)
        label_counts = {0: encrypted_label.count() - event_sum,
                        1: event_sum}
        sparse_bin_points = self.binning_obj.get_sparse_bin(self.bin_inner_param.bin_indexes,
                                                            self.binning_obj.split_points)
        sparse_bin_points = {self.bin_inner_param.header[k]: v for k, v in sparse_bin_points.items()}

        f = functools.partial(self.binning_obj.add_label_in_partition,
                              sparse_bin_points=sparse_bin_points)

        encrypted_bin_sum = data_bin_with_label.mapReducePartitions(f, self.binning_obj.aggregate_partition_label)

        def cal_zeros(bin_results):
            for b in bin_results:
                b[1] = b[1] - b[0]
            return bin_results

        encrypted_bin_sum = encrypted_bin_sum.mapValues(cal_zeros)

        f = functools.partial(self.binning_obj.fill_sparse_result,
                              sparse_bin_points=sparse_bin_points,
                              label_counts=label_counts)
        encrypted_bin_sum = encrypted_bin_sum.map(f)

        return encrypted_bin_sum

    def cipher_compress(self, encrypted_bin_sum, max_value):
        encrypted_bin_sum = encrypted_bin_sum.map(self.convert_compress_format)

        def _compress(col_dict):
            cipher_max_int = None
            res = {}
            event_counts = col_dict.get("event_counts")
            for v in event_counts:
                if isinstance(v, PaillierEncryptedNumber):
                    cipher_max_int = v.public_key.max_int
                    break
            if cipher_max_int is None:
                raise ValueError("All event counts are 0, please check data input.")
            _compressor = compressor.CipherCompressor(consts.PAILLIER, max_value,
                                                      cipher_max_int, compressor.NormalCipherPackage, 0)
            res["event_counts"] = _compressor.compress(col_dict["event_counts"])
            res["non_event_counts"] = _compressor.compress(col_dict["non_event_counts"])
            return res

        converted_bin_sum = encrypted_bin_sum.mapValues(_compress)
        return converted_bin_sum

    @staticmethod
    def convert_compress_format(col_name, encrypted_bin_sum):
        """
        Parameters
        ----------
        encrypted_bin_sum :  list.
            It is like:
                {'x1': [[event_count, non_event_count], [event_count, non_event_count] ... ],
                 'x2': [[event_count, non_event_count], [event_count, non_event_count] ... ],
                 ...
                }

        returns
        -------
        {"keys": ['x1', 'x2' ...],
         "event_counts": [...],
         "non_event_counts": [...],
         "bin_num": [...]
         }
        """
        event_counts = [x[0] for x in encrypted_bin_sum]
        non_event_counts = [x[1] for x in encrypted_bin_sum]
        return col_name, {"event_counts": event_counts, "non_event_counts": non_event_counts}

    def optimal_binning_sync(self):
        bucket_idx = self.transfer_variable.bucket_idx.get(idx=0)
        LOGGER.debug("In optimal_binning_sync, received bucket_idx: {}".format(bucket_idx))
        original_split_points = self.binning_obj.bin_results.all_split_points
        for encoded_col_name, b_idx in bucket_idx.items():
            col_name = self.bin_inner_param.decode_col_name(encoded_col_name)
            ori_sp_list = original_split_points.get(col_name)
            optimal_result = [ori_sp_list[i] for i in b_idx]
            self.binning_obj.bin_results.put_col_split_points(col_name, optimal_result)
