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
#  added by jsweng
#  alignemnet arbiter

from collections import defaultdict

from federatedml.feature.homo_onehot.homo_ohe_base import HomoOneHotBase
from federatedml.util import LOGGER
from federatedml.util import consts


class HomoOneHotArbiter(HomoOneHotBase):
    def __init__(self):
        super(HomoOneHotArbiter, self).__init__()

    def combine_all_column_headers(self, guest_columns, host_columns):
        """ This is used when there is a need for aligment within the
        federated learning. The function would align the column headers from
        guest and host and send the new aligned headers back.

        Returns:
            Combine all the column headers from guest and host
            if there is alignment is used
        """
        all_cols_dict = defaultdict(list)

        # Obtain all the guest headers
        for guest_cols in guest_columns:
            for k, v in guest_cols.items():
                all_cols_dict[k] = list(set(all_cols_dict[k] + v))

        # Obtain all the host headers
        for host_cols in host_columns:
            for k, v in host_cols.items():
                all_cols_dict[k] = list(set(all_cols_dict[k] + v))

        # Align all of them together
        combined_all_cols = {}
        for el in all_cols_dict.keys():
            combined_all_cols[el] = all_cols_dict[el]

        LOGGER.debug("{} combined cols: {}".format(self.role, combined_all_cols))

        return combined_all_cols

    def fit(self, data_instances=None):

        if self.need_alignment:
            guest_columns = self.transfer_variable.guest_columns.get(idx=-1)  # getting guest column
            host_columns = self.transfer_variable.host_columns.get(idx=-1)  # getting host column

            combined_all_cols = self.combine_all_column_headers(guest_columns, host_columns)

            # Send the aligned headers back to guest and host
            self.transfer_variable.aligned_columns.remote(combined_all_cols, role=consts.HOST, idx=-1)
            self.transfer_variable.aligned_columns.remote(combined_all_cols, role=consts.GUEST, idx=-1)

    def _get_meta(self):
        pass

    def _get_param(self):
        pass

    def export_model(self):

        return None

    def _load_model(self, model_dict):
        pass

    def transform(self, data_instances):
        pass

    def load_model(self, model_dict):
        pass
