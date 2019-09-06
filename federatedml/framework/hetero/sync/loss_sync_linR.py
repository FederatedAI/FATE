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

from federatedml.util import consts


class Arbiter(object):
    def _register_loss_sync(self, loss_transfer):
        self.loss_transfer = loss_transfer

    def sync_loss_info(self, n_iter_, batch_index):
        current_suffix = (n_iter_, batch_index)
        loss = self.loss_transfer.get(idx=0, suffix=current_suffix)
        return loss


class Guest(object):
    def _register_loss_sync(self, host_forward_loss_transfer, loss_transfer):
        self.host_forward_loss_transfer = host_forward_loss_transfer
        self.loss_transfer = loss_transfer

    def sync_loss_info(self, lr_variables, loss, n_iter_, batch_index, optimizer):
        current_suffix = (n_iter_, batch_index)

        guest_regular_loss = optimizer.loss_norm(lr_variables.coef_)
        if guest_regular_loss is not None:
            en_host_forward_loss = self.host_forward_loss_transfer.get(idx=0, suffix=current_suffix)
            loss += guest_regular_loss
            loss += en_host_forward_loss

        self.loss_transfer.remote(loss, role=consts.ARBITER, idx=0, suffix=current_suffix)


class Host(object):
    def _register_loss_sync(self, host_forward_loss_transfer, loss_transfer):
        self.host_forward_loss_transfer = host_forward_loss_transfer
        self.loss_transfer = loss_transfer

    def sync_loss_info(self, lr_variables, loss, n_iter_, batch_index, cipher, optimizer):
        current_suffix = (n_iter_, batch_index)

        host_regular_loss = optimizer.loss_norm(lr_variables.coef_)
        if host_regular_loss is not None:
            en_forward_loss = cipher.encrypt(loss + host_regular_loss)
            self.host_forward_loss_transfer.remote(en_forward_loss, role=consts.GUEST, idx=0, suffix=current_suffix)
