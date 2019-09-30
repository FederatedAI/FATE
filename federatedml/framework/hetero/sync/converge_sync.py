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


from federatedml.util import consts


class Arbiter(object):
    # noinspection PyAttributeOutsideInit
    def _register_convergence(self, is_stopped_transfer):
        self._is_stopped_transfer = is_stopped_transfer

    def sync_converge_info(self, is_converged, suffix=tuple()):
        self._is_stopped_transfer.remote(obj=is_converged, role=consts.HOST, idx=-1, suffix=suffix)
        self._is_stopped_transfer.remote(obj=is_converged, role=consts.GUEST, idx=-1, suffix=suffix)


class _Client(object):
    # noinspection PyAttributeOutsideInit
    def _register_convergence(self, is_stopped_transfer):
        self._is_stopped_transfer = is_stopped_transfer

    def sync_converge_info(self, suffix=tuple()):
        is_converged = self._is_stopped_transfer.get(idx=0, suffix=suffix)
        return is_converged


Host = _Client
Guest = _Client
