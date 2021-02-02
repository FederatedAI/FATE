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
import numpy

from federatedml.util import LOGGER
from federatedml.transfer_variable.transfer_class.feldman_verifiable_sum_transfer_variable import \
    FeldmanVerifiableSumTransferVariables
from federatedml.param.feldman_verifiable_sum_param import FeldmanVerifiableSumParam
from federatedml.statistic.feldman_verifiable_sum.base_feldman_verifiable_sum import BaseFeldmanVerifiableSum


class FeldmanVerifiableSumHost(BaseFeldmanVerifiableSum):
    def __init__(self):
        super(FeldmanVerifiableSumHost, self).__init__()
        self.transfer_inst = FeldmanVerifiableSumTransferVariables()
        self.host_party_idlist = []
        self.local_partyid = -1

    def _init_model(self, model_param: FeldmanVerifiableSumParam):
        self.sum_cols = model_param.sum_cols
        self.vss.Q_n = model_param.q_n

    def _init_data(self, data_inst):
        self.local_partyid = self.component_properties.local_partyid
        self.host_party_idlist = self.component_properties.host_party_idlist
        self.host_count = len(self.host_party_idlist)
        self.vss.key_pair()
        self.vss.set_share_amount(self.host_count)
        if not self.sum_cols:
            self.x = data_inst.mapValues(lambda x: x.features)
        else:
            self.x = data_inst.mapValues(self.select_data_by_idx)

    def select_data_by_idx(self, values):
        data = []
        for idx, feature in enumerate(values.features):
            if idx in self.model_param.sum_cols:
                data.append(feature)
        return numpy.array(data)

    def sync_share_to_parties(self):
        for idx, party_id in enumerate(self.host_party_idlist):
            if self.local_partyid != party_id:
                self.transfer_inst.host_share_to_host.remote(self.sub_key[idx],
                                                             role="host",
                                                             idx=idx)
            else:
                self.x_plus_y = self.sub_key[idx]
                self.transfer_inst.host_share_to_guest.remote(self.sub_key[-1],
                                                              role="guest",
                                                              idx=0)
        self.transfer_inst.host_commitments.remote(self.commitments, role="host", idx=-1)
        self.transfer_inst.host_commitments.remote(self.commitments, role="guest", idx=-1)

    def recv_share_from_parties(self):
        for idx, party_id in enumerate(self.host_party_idlist):
            if self.local_partyid != party_id:
                sub_key = self.transfer_inst.host_share_to_host.get(idx=idx)
                commitment = self.transfer_inst.host_commitments.get(idx=idx)
                self.verify_subkey(sub_key, commitment, self.component_properties.host_party_idlist[idx])
                self.y_recv.append(sub_key)
            else:
                sub_key = self.transfer_inst.guest_share_subkey.get(idx=0)
                commitment = self.transfer_inst.guest_commitments.get(idx=0)
                self.verify_subkey(sub_key, commitment, self.component_properties.guest_partyid)
                self.y_recv.append(sub_key)

    def sync_host_sum_to_guest(self):
        self.transfer_inst.host_sum.remote(self.x_plus_y,
                                           role="guest",
                                           idx=-1)

    def fit(self, data_inst):

        LOGGER.info("begin to make host data")
        self._init_data(data_inst)

        LOGGER.info("split data into multiple random parts")
        self.secure()

        LOGGER.info("share one of random part data to multiple parties")
        self.sync_share_to_parties()

        LOGGER.info("get share of one random part data from multiple parties")
        self.recv_share_from_parties()

        LOGGER.info("begin to get sum of multiple party")
        self.sub_key_sum()

        LOGGER.info("send host sum to guest")
        self.sync_host_sum_to_guest()
