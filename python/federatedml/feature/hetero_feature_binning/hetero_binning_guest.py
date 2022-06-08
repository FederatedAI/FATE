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

import copy
import functools

import numpy as np

from federatedml.cipher_compressor.packer import GuestIntegerPacker
from federatedml.feature.binning.iv_calculator import IvCalculator
from federatedml.secureprotol.encrypt_mode import EncryptModeCalculator
from federatedml.feature.binning.optimal_binning.optimal_binning import OptimalBinning
from federatedml.feature.hetero_feature_binning.base_feature_binning import BaseFeatureBinning
from federatedml.secureprotol import PaillierEncrypt
from federatedml.secureprotol.fate_paillier import PaillierEncryptedNumber
from federatedml.statistic import data_overview
from federatedml.statistic import statics
from federatedml.util import LOGGER
from federatedml.util import consts


class HeteroFeatureBinningGuest(BaseFeatureBinning):

    def __init__(self):
        super().__init__()
        self._packer: GuestIntegerPacker = None

    def fit(self, data_instances):
        """
        Apply binning method for both data instances in local party as well as the other one. Afterwards, calculate
        the specific metric value for specific columns. Currently, iv is support for binary labeled data only.
        """
        LOGGER.info("Start feature binning fit and transform")
        self._abnormal_detection(data_instances)

        # self._parse_cols(data_instances)

        self._setup_bin_inner_param(data_instances, self.model_param)

        if self.model_param.method == consts.OPTIMAL:
            has_missing_value = self.iv_calculator.check_containing_missing_value(data_instances)
            for idx in self.bin_inner_param.bin_indexes:
                if idx in has_missing_value:
                    raise ValueError(f"Optimal Binning do not support missing value now.")
        split_points = self.binning_obj.fit_split_points(data_instances)

        if self.model_param.skip_static:
            self.transform(data_instances)
            return self.data_output

        label_counts_dict = data_overview.get_label_count(data_instances)

        if len(label_counts_dict) > 2:
            if self.model_param.method == consts.OPTIMAL:
                raise ValueError("Have not supported optimal binning in multi-class data yet")

        self.labels = list(label_counts_dict.keys())
        label_counts = [label_counts_dict[k] for k in self.labels]
        label_table = IvCalculator.convert_label(data_instances, self.labels)
        self.bin_result = self.iv_calculator.cal_local_iv(data_instances=data_instances,
                                                          split_points=split_points,
                                                          labels=self.labels,
                                                          label_counts=label_counts,
                                                          bin_cols_map=self.bin_inner_param.get_need_cal_iv_cols_map(),
                                                          label_table=label_table)

        if self.model_param.local_only:

            self.transform(data_instances)
            self.set_summary(self.bin_result.summary())
            return self.data_output

        if self.model_param.encrypt_param.method == consts.PAILLIER:
            paillier_encryptor = PaillierEncrypt()
            paillier_encryptor.generate_key(self.model_param.encrypt_param.key_length)
        else:
            raise NotImplementedError("encrypt method not supported yet")
        self._packer = GuestIntegerPacker(pack_num=len(self.labels), pack_num_range=label_counts,
                                          encrypter=paillier_encryptor)

        self.federated_iv(data_instances=data_instances, label_table=label_table,
                          result_counts=label_counts_dict, label_elements=self.labels)

        total_summary = self.bin_result.summary()
        for host_res in self.host_results:
            total_summary = self._merge_summary(total_summary, host_res.summary())

        self.set_schema(data_instances)
        self.transform(data_instances)
        LOGGER.info("Finish feature binning fit and transform")
        self.set_summary(total_summary)
        return self.data_output

    def federated_iv(self, data_instances, label_table, result_counts, label_elements):

        converted_label_table = label_table.mapValues(lambda x: [int(i) for i in x])
        encrypted_label_table = self._packer.pack_and_encrypt(converted_label_table)
        self.transfer_variable.encrypted_label.remote(encrypted_label_table,
                                                      role=consts.HOST,
                                                      idx=-1)
        encrypted_bin_sum_infos = self.transfer_variable.encrypted_bin_sum.get(idx=-1)
        encrypted_bin_infos = self.transfer_variable.optimal_info.get(idx=-1)
        LOGGER.info("Get encrypted_bin_sum from host")
        for host_idx, encrypted_bin_info in enumerate(encrypted_bin_infos):
            host_party_id = self.component_properties.host_party_idlist[host_idx]
            encrypted_bin_sum = encrypted_bin_sum_infos[host_idx]
            # assert 1 == 2, f"encrypted_bin_sum: {list(encrypted_bin_sum.collect())}"
            result_counts_table = self._packer.decrypt_cipher_package_and_unpack(encrypted_bin_sum)
            # LOGGER.debug(f"unpack result: {result_counts_table.first()}")

            bin_result = self.cal_bin_results(data_instances=data_instances,
                                              host_idx=host_idx,
                                              encrypted_bin_info=encrypted_bin_info,
                                              result_counts_table=result_counts_table,
                                              result_counts=result_counts,
                                              label_elements=label_elements)
            bin_result.set_role_party(role=consts.HOST, party_id=host_party_id)
            self.host_results.append(bin_result)

    def host_optimal_binning(self, data_instances, host_idx, encrypted_bin_info, result_counts, category_names):
        optimal_binning_params = encrypted_bin_info['optimal_params']

        host_model_params = copy.deepcopy(self.model_param)
        host_model_params.bin_num = optimal_binning_params.get('bin_num')
        host_model_params.optimal_binning_param.metric_method = optimal_binning_params.get('metric_method')
        host_model_params.optimal_binning_param.mixture = optimal_binning_params.get('mixture')
        host_model_params.optimal_binning_param.max_bin_pct = optimal_binning_params.get('max_bin_pct')
        host_model_params.optimal_binning_param.min_bin_pct = optimal_binning_params.get('min_bin_pct')

        event_total, non_event_total = self.get_histogram(data_instances)
        result_counts = dict(result_counts.collect())
        optimal_binning_cols = {x: y for x, y in result_counts.items() if x not in category_names}
        host_binning_obj = OptimalBinning(params=host_model_params, abnormal_list=self.binning_obj.abnormal_list)
        host_binning_obj.event_total = event_total
        host_binning_obj.non_event_total = non_event_total
        host_binning_obj = self.optimal_binning_sync(host_binning_obj, optimal_binning_cols, data_instances.count(),
                                                     data_instances.partitions,
                                                     host_idx)
        return host_binning_obj

    def cal_bin_results(self, data_instances, host_idx, encrypted_bin_info, result_counts_table,
                        result_counts, label_elements):
        host_bin_methods = encrypted_bin_info['bin_method']
        category_names = encrypted_bin_info['category_names']
        result_counts_dict = dict(result_counts_table.collect())
        host_party_id = self.component_properties.host_party_idlist[host_idx]
        if host_bin_methods == consts.OPTIMAL:
            if len(result_counts) > 2:
                raise ValueError("Have not supported optimal binning in multi-class data yet")
            host_binning_obj = self.host_optimal_binning(data_instances, host_idx,
                                                         encrypted_bin_info, result_counts_table,
                                                         category_names)
            optimal_counts = {}
            for col_name, bucket_list in host_binning_obj.bucket_lists.items():
                optimal_counts[col_name] = [np.array([b.event_count, b.non_event_count]) for b in bucket_list]

            for col_name, counts in result_counts_dict.items():
                if col_name in category_names:
                    optimal_counts[col_name] = counts
            # LOGGER.debug(f"optimal_counts: {optimal_counts}")
            bin_res = self.iv_calculator.cal_iv_from_counts(optimal_counts, labels=label_elements,
                                                            role=consts.HOST, party_id=host_party_id)
        else:
            bin_res = self.iv_calculator.cal_iv_from_counts(result_counts_table,
                                                            label_elements,
                                                            role=consts.HOST,
                                                            party_id=host_party_id)
        return bin_res

    @staticmethod
    def convert_decompress_format(encrypted_bin_sum):
        """
        Parameters
        ----------
        encrypted_bin_sum :  dict.
            {"keys": ['x1', 'x2' ...],
             "event_counts": [...],
             "non_event_counts": [...],
             bin_num": [...]
            }
        returns
        -------
                {'x1': [[event_count, non_event_count], [event_count, non_event_count] ... ],
                 'x2': [[event_count, non_event_count], [event_count, non_event_count] ... ],
                 ...
                }
        """
        result = {}
        start = 0
        event_counts = [int(x) for x in encrypted_bin_sum['event_counts']]
        non_event_counts = [int(x) for x in encrypted_bin_sum['non_event_counts']]
        for idx, k in enumerate(encrypted_bin_sum["keys"]):
            bin_num = encrypted_bin_sum["bin_nums"][idx]
            result[k] = list(zip(event_counts[start: start + bin_num], non_event_counts[start: start + bin_num]))
            start += bin_num
        assert start == len(event_counts) == len(non_event_counts), \
            f"Length of event/non-event does not match " \
            f"with bin_num sums, all_counts: {start}, length of event count: {len(event_counts)}," \
            f"length of non_event_counts: {len(non_event_counts)}"
        return result

    @staticmethod
    def _merge_summary(summary_1, summary_2):
        def merge_single_label(s1, s2):
            res = {}
            for k, v in s1.items():
                if k == 'iv':
                    v.extend(s2[k])
                    v = sorted(v, key=lambda p: p[1], reverse=True)
                else:
                    v.update(s2[k])
                res[k] = v
            return res

        res = {}
        for label, s1 in summary_1.items():
            s2 = summary_2.get(label)
            res[label] = merge_single_label(s1, s2)
        return res

    @staticmethod
    def encrypt(x, cipher):
        if not isinstance(x, np.ndarray):
            return cipher.encrypt(x)
        res = []
        for idx, value in enumerate(x):
            res.append(cipher.encrypt(value))
        return np.array(res)

    @staticmethod
    def __decrypt_bin_sum(encrypted_bin_sum, cipher):

        def decrypt(values):
            res = []
            for counts in values:
                for idx, c in enumerate(counts):
                    if isinstance(c, PaillierEncryptedNumber):
                        counts[idx] = cipher.decrypt(c)
                res.append(counts)
            return res
        return encrypted_bin_sum.mapValues(decrypt)

    @staticmethod
    def load_data(data_instance):
        data_instance = copy.deepcopy(data_instance)
        # Here suppose this is a binary question and the event label is 1
        if data_instance.label != 1:
            data_instance.label = 0
        return data_instance

    def optimal_binning_sync(self, host_binning_obj, result_counts, sample_count, partitions, host_idx):
        LOGGER.debug("Start host party optimal binning train")
        bucket_table = host_binning_obj.bin_sum_to_bucket_list(result_counts, partitions)
        host_binning_obj.fit_buckets(bucket_table, sample_count)
        encoded_split_points = host_binning_obj.bin_results.all_split_points
        self.transfer_variable.bucket_idx.remote(encoded_split_points,
                                                 role=consts.HOST,
                                                 idx=host_idx)
        return host_binning_obj

    @staticmethod
    def get_histogram(data_instances):
        static_obj = statics.MultivariateStatisticalSummary(data_instances, cols_index=-1)
        label_historgram = static_obj.get_label_histogram()
        event_total = label_historgram.get(1, 0)
        non_event_total = label_historgram.get(0, 0)
        if event_total == 0 or non_event_total == 0:
            LOGGER.warning(f"event_total or non_event_total might have errors, event_total: {event_total},"
                           f" non_event_total: {non_event_total}")
        return event_total, non_event_total
