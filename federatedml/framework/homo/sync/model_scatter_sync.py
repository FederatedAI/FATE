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

from federatedml.framework.weights import TransferableWeights
from federatedml.framework.homo.util import scatter
from federatedml.util import consts


class Arbiter(object):

    # noinspection PyAttributeOutsideInit
    def register_model_scatter(self, host_model_transfer, guest_model_transfer):
        self._models_sync = scatter.Scatter(host_model_transfer, guest_model_transfer)
        return self

    def get_models(self, ciphers_dict=None, suffix=tuple()):

        # guest model
        models_iter = self._models_sync.get(suffix=suffix)
        guest_model = next(models_iter)
        yield (guest_model.weights, guest_model.get_degree() or 1.0)

        # host model
        index = 0
        for model in models_iter:
            weights = model.weights
            if ciphers_dict and ciphers_dict.get(index, None):
                weights = weights.decrypted(ciphers_dict[index])
            yield (weights, model.get_degree() or 1.0)
            index += 1


class _Client(object):
    # noinspection PyAttributeOutsideInit
    def register_model_scatter(self, model_transfer):
        self._models_sync = model_transfer
        return self

    def send_model(self, weights: TransferableWeights, suffix=tuple()):
        self._models_sync.remote(obj=weights, role=consts.ARBITER, idx=0, suffix=suffix)
        return weights


Guest = _Client
Host = _Client
