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

from arch.api.utils.splitable import segment_transfer_enabled
from federatedml.secureprotol.encrypt import Encrypt, FakeEncrypt


class TransferableWeights(metaclass=segment_transfer_enabled()):

    def __init__(self, d):
        self._weights = d

    def update(self, other):
        for k, v in other.items():
            self._weights[k] = v

    def encrypted(self, cipher: Encrypt, inplace=True):
        if inplace:
            if isinstance(cipher, FakeEncrypt):
                return
            for k, v in self._weights.items():
                self._weights[k] = cipher.encrypt(v)
            return self
        else:
            _w = dict()
            for k, v in self._weights.items():
                _w[k] = cipher.encrypt(v)
            return TransferableWeights(_w)

    def __getitem__(self, item):
        return self._weights[item]

    def __setitem__(self, key, value):
        self._weights[key] = value

    def __contains__(self, item):
        return self._weights.__contains__(item)

    def items(self):
        return self._weights.items()

    def keys(self):
        return self._weights.keys()
