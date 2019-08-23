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

from federatedml.homo.weights import Parameters
from federatedml.homo.transfer import arbiter_broadcast


def _tag_suffix(version):
    return f"version_{version}"


class _Arbiter(object):
    def __init__(self, dispatcher):
        self._dispatcher = dispatcher

    def send(self, model_weights: Parameters, version, ciphers):
        if ciphers:
            for idx, cipher in ciphers.items():
                encrypt_model = model_weights.encrypted(cipher, inplace=False)
                self._dispatcher.remote(encrypt_model.for_remote(), suffix=_tag_suffix(version), idx=idx)
        else:
            self._dispatcher.remote(model_weights, suffix=_tag_suffix(version), idx=-1)


class _Host(object):
    def __init__(self, dispatcher):
        self._dispatcher = dispatcher

    def get(self, version):
        return self._dispatcher.get(suffix=_tag_suffix(version))


class _Guest(object):
    def __init__(self, dispatcher):
        self._dispatcher = dispatcher

    def get(self, version):
        return self._dispatcher.get(suffix=_tag_suffix(version))


def _parse_transfer_variable(transfer_variable):
    return arbiter_broadcast(name=transfer_variable.aggregated.name,
                             tag=transfer_variable.generate_transferid(transfer_variable.mean_gradient))


class AggregatedDisPatcher(object):
    """@Arbiter -> [@Host, @Guest]
    transfer model from arbiter to hosts and guest
    """

    @staticmethod
    def arbiter(transfer_variable):
        return _Arbiter(_parse_transfer_variable(transfer_variable))

    @staticmethod
    def host(transfer_variable):
        return _Host(_parse_transfer_variable(transfer_variable))

    @staticmethod
    def guest(transfer_variable):
        return _Guest(_parse_transfer_variable(transfer_variable))
