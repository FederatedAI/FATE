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

from federatedml.framework.weights import Variables
from federatedml.framework.homo.util import scatter
from federatedml.util import consts


class Arbiter(object):

    # noinspection PyAttributeOutsideInit
    def _register_model_scatter(self, host_model_transfer, guest_model_transfer):
        self._models_sync = scatter.Scatter(host_model_transfer, guest_model_transfer)
        return self

    def _get_models(self, ciphers_dict=None, suffix=tuple()):
        models = [Variables.from_transferable(model) for model in self._models_sync.get(suffix=suffix)]
        if ciphers_dict:
            for i, cipher in ciphers_dict.items():
                if cipher:
                    models[i + 1] = models[i + 1].decrypted(ciphers_dict[i])
        return models


class _Client(object):
    # noinspection PyAttributeOutsideInit
    def _register_model_scatter(self, model_transfer):
        self._models_sync = model_transfer
        return self

    def _send_model(self, weights: Variables, suffix=tuple()):
        self._models_sync.remote(obj=weights.for_remote(), role=consts.ARBITER, idx=0, suffix=suffix)
        return weights


Guest = _Client
Host = _Client
