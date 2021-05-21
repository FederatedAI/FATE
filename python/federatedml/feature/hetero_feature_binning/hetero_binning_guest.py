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

from federatedml.cipher_compressor.compressor import CipherDecompressor
from federatedml.feature.binning.base_binning import BaseBinning
from federatedml.feature.binning.optimal_binning.optimal_binning import OptimalBinning
from federatedml.feature.hetero_feature_binning.base_feature_binning import BaseFeatureBinning
from federatedml.secureprotol import PaillierEncrypt
from federatedml.secureprotol.fate_paillier import PaillierEncryptedNumber
from federatedml.statistic import data_overview
from federatedml.statistic import statics
from federatedml.util import LOGGER
from federatedml.util import consts


class HeteroFeatureBinningGuest(BaseFeatureBinning):
    def fit(self, data_instances):
        """
        Apply binning method for both data instances in local party as well as the other one. Afterwards, calculate
        the specific metric value for specific columns. Currently, iv is support for binary labeled data only.
        """
        LOGGER.info("Start feature binning fit and transform")
        self._abnormal_detection(data_instances)

        # self._parse_cols(data_instances)
        self._setup_bin_inner_param(data_instances, self.model_param)

        self.binning_obj.fit_split_points(data_instances)
        if self.model_param.skip_static:
            self.transform(data_instances)
            return self.data_output

        label_counts = data_overview.get_label_count(data_instances)
        if len(label_counts) > 2:
            raise ValueError("Iv calculation support binary-data only in this version.")

        data_instances = data_instances.mapValues(self.load_data)
        self.set_schema(data_instances)
        label_table = data_instances.mapValues(lambda x: x.label)

        if self.model_param.local_only:
            LOGGER.info("This is a local only binning fit")
            self.binning_obj.cal_local_iv(data_instances, label_table=label_table,
                                          label_counts=label_counts)
            self.transform(data_instances)
            self.set_summary(self.binning_obj.bin_results.summary())
            return self.data_output

        if self.model_param.encrypt_param.method == consts.PAILLIER:
            cipher = PaillierEncrypt()
            cipher.generate_key(self.model_param.encrypt_param.key_length)
        else:
            raise NotImplementedError("encrypt method not supported yet")
        # from federatedml.secureprotol.encrypt import FakeEncrypt
        # cipher = FakeEncrypt()
        f = functools.partial(self.encrypt, cipher=cipher)
        encrypted_label_table = label_table.mapValues(f)

        self.transfer_variable.encrypted_label.remote(encrypted_label_table,
                                                      role=consts.HOST,
                                                      idx=-1)
        LOGGER.info("Sent encrypted_label_table to host")

        self.binning_obj.cal_local_iv(data_instances, label_table=label_table,
                                      label_counts=label_counts)

        encrypted_bin_sum_infos = self.transfer_variable.encrypted_bin_sum.get(idx=-1)
        encrypted_bin_infos = self.transfer_variable.optimal_info.get(idx=-1)
        total_summary = self.binning_obj.bin_results.summary()

        LOGGER.info("Get encrypted_bin_sum from host")
        for host_idx, encrypted_bin_info in enumerate(encrypted_bin_infos):
            host_party_id = self.component_properties.host_party_idlist[host_idx]
            encrypted_bin_sum = encrypted_bin_sum_infos[host_idx]
            result_counts = self.cipher_decompress(encrypted_bin_sum, cipher)

            host_bin_methods = encrypted_bin_info['bin_method']
            category_names = encrypted_bin_info['category_names']
            if host_bin_methods == consts.OPTIMAL:
                optimal_binning_params = encrypted_bin_info['optimal_params']

                host_model_params = copy.deepcopy(self.model_param)
                host_model_params.bin_num = optimal_binning_params.get('bin_num')
                host_model_params.optimal_binning_param.metric_method = optimal_binning_params.get('metric_method')
                host_model_params.optimal_binning_param.mixture = optimal_binning_params.get('mixture')
                host_model_params.optimal_binning_param.max_bin_pct = optimal_binning_params.get('max_bin_pct')
                host_model_params.optimal_binning_param.min_bin_pct = optimal_binning_params.get('min_bin_pct')

                self.binning_obj.event_total, self.binning_obj.non_event_total = self.get_histogram(data_instances)
                result_counts = dict(result_counts.collect())
                optimal_binning_cols = {x: y for x, y in result_counts.items() if x not in category_names}
                host_binning_obj = self.optimal_binning_sync(optimal_binning_cols, data_instances.count(),
                                                             data_instances.partitions,
                                                             host_idx, host_model_params)
                category_bins = {x: y for x, y in result_counts.items() if x in category_names}
                host_binning_obj.cal_iv_woe(category_bins, self.model_param.adjustment_factor)
            else:
                host_binning_obj = BaseBinning()
                host_binning_obj.cal_iv_woe(result_counts, self.model_param.adjustment_factor)
            host_binning_obj.set_role_party(role=consts.HOST, party_id=host_party_id)
            total_summary = self._merge_summary(total_summary,
                                                host_binning_obj.bin_results.summary())
            self.host_results.append(host_binning_obj)

        self.set_schema(data_instances)
        self.transform(data_instances)
        LOGGER.info("Finish feature binning fit and transform")
        total_summary['test'] = 'test'
        self.set_summary(total_summary)
        return self.data_output

    def cipher_decompress(self, encrypted_bin_sum, cipher):

        def _decompress(col_dict):
            _decompressor = CipherDecompressor(encrypter=cipher)
            event_counts = _decompressor.unpack(col_dict["event_counts"])
            event_counts = [int(x) for x in event_counts]
            non_event_counts = _decompressor.unpack(col_dict["non_event_counts"])
            non_event_counts = [int(x) for x in non_event_counts]
            res = list(zip(event_counts, non_event_counts))
            return res

        encrypted_bin_sum = encrypted_bin_sum.mapValues(_decompress)
        # return self.convert_decompress_format(encrypted_bin_sum)
        return encrypted_bin_sum

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
        res = {}
        for k, v in summary_1.items():
            if k == 'iv':
                v.extend(summary_2[k])
                v = sorted(v, key=lambda p: p[1], reverse=True)
            else:
                v.update(summary_2[k])
            res[k] = v

        return res

    @staticmethod
    def encrypt(x, cipher):
        return cipher.encrypt(x)

    @staticmethod
    def __decrypt_bin_sum(encrypted_bin_sum, cipher):
        # for feature_sum in encrypted_bin_sum:
        decrypted_list = {}
        for col_name, count_list in encrypted_bin_sum.items():
            new_list = []
            for event_count, non_event_count in count_list:
                if isinstance(event_count, PaillierEncryptedNumber):
                    event_count = cipher.decrypt(event_count)
                if isinstance(non_event_count, PaillierEncryptedNumber):
                    non_event_count = cipher.decrypt(non_event_count)
                new_list.append((event_count, non_event_count))
            decrypted_list[col_name] = new_list
        return decrypted_list

    @staticmethod
    def load_data(data_instance):
        data_instance = copy.deepcopy(data_instance)
        # Here suppose this is a binary question and the event label is 1
        if data_instance.label != 1:
            data_instance.label = 0
        return data_instance

    def optimal_binning_sync(self, result_counts, sample_count, partitions, host_idx, host_model_params):
        host_binning_obj = OptimalBinning(params=host_model_params, abnormal_list=self.binning_obj.abnormal_list)
        host_binning_obj.event_total = self.binning_obj.event_total
        host_binning_obj.non_event_total = self.binning_obj.non_event_total
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
