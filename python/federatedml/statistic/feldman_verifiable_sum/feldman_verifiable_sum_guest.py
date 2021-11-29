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

import copy
import numpy

from federatedml.util import LOGGER
from federatedml.feature.instance import Instance
from federatedml.transfer_variable.transfer_class import feldman_verifiable_sum_transfer_variable
from federatedml.param.feldman_verifiable_sum_param import FeldmanVerifiableSumParam
from federatedml.statistic.feldman_verifiable_sum.base_feldman_verifiable_sum import BaseFeldmanVerifiableSum


class FeldmanVerifiableSumGuest(BaseFeldmanVerifiableSum):
    def __init__(self):
        super(FeldmanVerifiableSumGuest, self).__init__()
        self.transfer_inst = feldman_verifiable_sum_transfer_variable.FeldmanVerifiableSumTransferVariables()
        self.output_schema = None

    def _init_model(self, model_param: FeldmanVerifiableSumParam):
        self.sum_cols = model_param.sum_cols
        self.vss.Q_n = model_param.q_n

    def _init_data(self, data_inst):
        self.host_count = len(self.component_properties.host_party_idlist)
        self.vss.key_pair()
        self.vss.set_share_amount(self.host_count)
        if not self.model_param.sum_cols:
            self.x = data_inst.mapValues(lambda x: x.features)
            self.output_schema = data_inst.schema
        else:
            self.x = data_inst.mapValues(self.select_data_by_idx)
            header = []
            for idx, label in enumerate(data_inst.schema.get('header')):
                if idx in self.sum_cols:
                    header.append(label)
            # self.output_schema = {"header": header, "sid_name": data_inst.schema.get('sid_name')}
            schema = copy.deepcopy(data_inst.schema)
            schema["header"] = header
            self.output_schema = schema

    def select_data_by_idx(self, values):
        data = []
        for idx, feature in enumerate(values.features):
            if idx in self.sum_cols:
                data.append(feature)
        return numpy.array(data)

    def sync_share_to_host(self):
        for idx in range(self.host_count):
            self.transfer_inst.guest_share_subkey.remote(self.sub_key[idx],
                                                         role="host",
                                                         idx=idx)
        self.transfer_inst.guest_commitments.remote(self.commitments,
                                                    role="host",
                                                    idx=-1)
        self.x_plus_y = self.sub_key[-1]

    def recv_share_from_host(self):
        for idx in range(self.host_count):
            sub_key = self.transfer_inst.host_share_to_guest.get(idx=idx)
            commitment = self.transfer_inst.host_commitments.get(idx=idx)

            self.verify_subkey(sub_key, commitment, self.component_properties.host_party_idlist[idx])
            self.y_recv.append(sub_key)
            self.commitments_recv.append(commitment)

    def recv_host_sum_from_host(self):
        for idx in range(self.host_count):
            host_sum = self.transfer_inst.host_sum.get(idx=idx)
            self.verify_sumkey(host_sum, self.commitments, self.component_properties.host_party_idlist[idx])
            self.host_sum_recv.append(host_sum)

    def fit(self, data_inst):
        LOGGER.info("begin to make guest data")
        self._init_data(data_inst)

        LOGGER.info("split data into multiple random parts")
        self.secure()

        LOGGER.info("share one random part data to multiple hosts")
        self.sync_share_to_host()

        LOGGER.info("get share of one random part data from multiple hosts")
        self.recv_share_from_host()

        LOGGER.info("begin to get sum of multiple party")
        self.sub_key_sum()

        LOGGER.info("receive host sum from host")
        self.recv_host_sum_from_host()

        self.reconstruct()

        LOGGER.info("success to calculate privacy sum")
        self.secret_sum = self.secret_sum.join(data_inst, lambda s, v: Instance(features=numpy.array(s),
                                                                                inst_id=v.inst_id))

        self.secret_sum.schema = self.output_schema

        data_output = self.secret_sum

        return data_output
