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

from federatedml.framework.homo.util import scatter
from federatedml.util import consts


class Arbiter(object):

    # noinspection PyAttributeOutsideInit
    def register_loss_transfer(self, host_loss_transfer, guest_loss_transfer):
        self._loss_sync = scatter.Scatter(host_loss_transfer, guest_loss_transfer)
        return self

    def get_losses(self, idx=None, suffix=tuple()):
        return self._loss_sync.get(host_ids=idx, suffix=suffix)


class _Client(object):

    # noinspection PyAttributeOutsideInit
    def register_loss_transfer(self, loss_transfer):
        self._loss_sync = loss_transfer
        return self

    def send_loss(self, loss, suffix=tuple()):
        self._loss_sync.remote(obj=loss, role=consts.ARBITER, idx=0, suffix=suffix)
        return loss


Guest = _Client
Host = _Client
