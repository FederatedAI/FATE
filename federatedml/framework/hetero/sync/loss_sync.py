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

    def sync_loss_info(self, suffix=tuple()):
        loss = self.loss_transfer.get(idx=0, suffix=suffix)
        return loss


class Guest(object):
    def _register_loss_sync(self, host_loss_regular_transfer, loss_transfer, loss_intermediate_transfer):
        self.host_loss_regular_transfer = host_loss_regular_transfer
        self.loss_transfer = loss_transfer
        self.loss_intermediate_transfer = loss_intermediate_transfer

    def sync_loss_info(self, loss, suffix=tuple()):
        self.loss_transfer.remote(loss, role=consts.ARBITER, idx=0, suffix=suffix)

    def get_host_loss_intermediate(self, suffix=tuple()):
        loss_intermediate = self.loss_intermediate_transfer.get(idx=-1, suffix=suffix)
        return loss_intermediate

    def get_host_loss_regular(self, suffix=tuple()):
        losses = self.host_loss_regular_transfer.get(idx=-1, suffix=suffix)
        return losses


class Host(object):
    def _register_loss_sync(self, host_loss_regular_transfer, loss_transfer, loss_intermediate_transfer):
        self.host_loss_regular_transfer = host_loss_regular_transfer
        self.loss_transfer = loss_transfer
        self.loss_intermediate_transfer = loss_intermediate_transfer

    def remote_loss_intermediate(self, loss_intermediate, suffix=tuple()):
        self.loss_intermediate_transfer.remote(obj=loss_intermediate, role=consts.GUEST, idx=0, suffix=suffix)

    def remote_loss_regular(self, loss_regular, suffix=tuple()):
        self.host_loss_regular_transfer.remote(obj=loss_regular, role=consts.GUEST, idx=0, suffix=suffix)
