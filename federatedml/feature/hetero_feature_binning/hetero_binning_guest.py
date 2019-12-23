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
from federatedml.feature.binning.base_binning import HostBaseBinning
from federatedml.feature.hetero_feature_binning.base_feature_binning import BaseHeteroFeatureBinning
from federatedml.secureprotol import PaillierEncrypt
from federatedml.statistic import data_overview
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class HeteroFeatureBinningGuest(BaseHeteroFeatureBinning):
    # def __init__(self):
    #     super(HeteroFeatureBinningGuest, self).__init__()
    #

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
        LOGGER.debug("After fit, binning_obj split_points: {}".format(self.binning_obj.split_points))

        is_binary_data = data_overview.is_binary_labels(data_instances)
        if not is_binary_data:
            # LOGGER.warning("Iv calculation support binary-data only in this version.")
            raise ValueError("Iv calculation support binary-data only in this version.")
            # return data_instances

        data_instances = data_instances.mapValues(self.load_data)
        self.set_schema(data_instances)
        label_table = data_instances.mapValues(lambda x: x.label)

        if self.model_param.local_only:
            LOGGER.info("This is a local only binning fit")
            self.binning_obj.cal_local_iv(data_instances, label_table=label_table)
            self.transform(data_instances)
            return self.data_output

        cipher = PaillierEncrypt()
        cipher.generate_key()

        f = functools.partial(self.encrypt, cipher=cipher)
        encrypted_label_table = label_table.mapValues(f)

        self.transfer_variable.encrypted_label.remote(encrypted_label_table,
                                                      role=consts.HOST,
                                                      idx=-1)
        LOGGER.info("Sent encrypted_label_table to host")

        self.binning_obj.cal_local_iv(data_instances, label_table=label_table)

        encrypted_bin_sums = self.transfer_variable.encrypted_bin_sum.get(idx=-1)

        LOGGER.info("Get encrypted_bin_sum from host")
        for host_idx, encrypted_bin_sum in enumerate(encrypted_bin_sums):
            host_party_id = self.component_properties.host_party_idlist[host_idx]
            host_binning_obj = HostBaseBinning()
            host_binning_obj.set_role_party(role=consts.HOST, party_id=host_party_id)
            result_counts = self.__decrypt_bin_sum(encrypted_bin_sum, cipher)
            host_binning_obj.cal_iv_woe(result_counts, self.model_param.adjustment_factor)
            self.host_results.append(host_binning_obj)

        self.set_schema(data_instances)
        self.transform(data_instances)
        LOGGER.info("Finish feature binning fit and transform")
        return self.data_output

    @staticmethod
    def encrypt(x, cipher):
        return cipher.encrypt(x), cipher.encrypt(1 - x)

    @staticmethod
    def __decrypt_bin_sum(encrypted_bin_sum, cipher):
        # for feature_sum in encrypted_bin_sum:
        decrypted_list = {}
        for col_name, count_list in encrypted_bin_sum.items():
            new_list = []
            for encrypted_event, encrypted_non_event in count_list:
                event_count = cipher.decrypt(encrypted_event)
                non_event_count = cipher.decrypt(encrypted_non_event)
                new_list.append((event_count, non_event_count))
            decrypted_list[col_name] = new_list
        return decrypted_list

    @staticmethod
    def load_data(data_instance):
        # Here suppose this is a binary question and the event label is 1
        if data_instance.label != 1:
            data_instance.label = 0
        return data_instance
