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

from arch.api.utils.log_utils import LoggerFactory
from federatedml.framework.hetero_lr_utils.sync import converge_sync

LOGGER = LoggerFactory.get_logger()


class Host(converge_sync.Host):

    def register_convergence(self, transfer_variables):
        self._register_convergence(is_stopped_transfer=transfer_variables.is_stopped)


class Guest(converge_sync.Guest):

    def register_convergence(self, transfer_variables):
        self._register_convergence(is_stopped_transfer=transfer_variables.is_stopped)


class Arbiter(converge_sync.Arbiter):

    def register_convergence(self, transfer_variables):
        self._register_convergence(is_stopped_transfer=transfer_variables.is_stopped)

