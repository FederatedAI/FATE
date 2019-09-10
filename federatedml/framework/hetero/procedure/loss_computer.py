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


from arch.api.utils import log_utils
from federatedml.framework.hetero.sync import loss_sync

LOGGER = log_utils.getLogger()




class Arbiter(loss_sync.Arbiter):
    def register_loss_computer(self, transfer_variable):
        self._register_loss_sync(transfer_variable.loss)




class Guest(loss_sync.Guest):
    def register_loss_computer(self, transfer_variable):
        self._register_loss_sync(transfer_variable.host_loss_regular,
                                 transfer_variable.loss)

    @staticmethod
    def __compute_loss(values):
        """
        Compute hetero-lr loss for:
        loss = log2 - 1/2*ywx + 1/8*(wx)^2, where y is label, w is model weight and x is features
        Parameters
        ----------
        values: DTable, include 1/2*ywx and (wx)^2

        numpy.ndarray
            hetero-lr loss
        """
        bias = np.log(2)

        loss = 0
        counter = 0
        for _, value in values:
            l = value[0] * (-1) + value[1] / 8 + bias
            loss = loss + l
            counter += 1

        return np.array([loss, counter])

    def comute_loss(self):

        # compute and loss
        half_ywx = encrypted_wx.join(data_instance, lambda wx, d: 0.5 * wx * int(d.label))
        half_ywx_join_en_sum_wx_square = half_ywx.join(en_sum_wx_square, lambda yz, ez: (yz, ez))
        f = functools.partial(self.__compute_loss)
        loss_partition = half_ywx_join_en_sum_wx_square.mapPartitions(f).reduce(lambda x, y: x + y)
        loss = loss_partition[0] / loss_partition[1]


class Host(loss_sync.Host):
    def register_loss_computer(self, transfer_variable):
        self._register_loss_sync(transfer_variable.host_loss_regular,
                                 transfer_variable.loss)
