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
    def _register_loss_sync(self, host_loss_regular_transfer, loss_transfer, wx_square_transfer):
        self.host_loss_regular_transfer = host_loss_regular_transfer
        self.loss_transfer = loss_transfer
        self.wx_square_transfer = wx_square_transfer

    def sync_loss_info(self, loss, suffix=tuple()):
        self.loss_transfer.remote(loss, role=consts.ARBITER, idx=0, suffix=suffix)

    def get_host_wx_square(self, suffix=tuple()):
        wx_squares = self.wx_square_transfer.get(idx=-1, suffix=suffix)
        return wx_squares

    def get_host_loss(self, suffix=tuple()):
        losses = self.loss_transfer.get(idx=-1, suffix=suffix)
        return losses


class Host(object):
    def _register_loss_sync(self, host_loss_regular_transfer, loss_transfer, wx_square_transfer):
        self.host_loss_regular_transfer = host_loss_regular_transfer
        self.loss_transfer = loss_transfer
        self.wx_square_transfer = wx_square_transfer

    def remote_wx_square(self, wx_square, suffix=tuple()):
        self.wx_square_transfer.remote(obj=wx_square, role=consts.GUEST, idx=0, suffix=suffix)

    def remote_loss(self, loss, suffix=tuple()):
        self.loss_transfer.remote(obj=loss, role=consts.GUEST, idx=0, suffix=suffix)
