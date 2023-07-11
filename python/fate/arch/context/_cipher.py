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

import logging

from ..unify import device

logger = logging.getLogger(__name__)


class CipherKit:
    def __init__(self, device: device, cipher_mapping=None) -> None:
        self._device = device
        self._cipher_mapping = cipher_mapping

    @property
    def phe(self):
        if self._cipher_mapping is None:
            if self._device == device.CPU:
                return PHECipher("paillier")
            else:
                logger.warning(f"no impl exists for device {self._device}, fallback to CPU")
                return PHECipher("paillier")

        if "phe" not in self._cipher_mapping:
            raise ValueError("phe is not set")

        if self._device not in self._cipher_mapping["phe"]:
            raise ValueError(f"phe is not set for device {self._device}")

        return PHECipher(self._cipher_mapping["phe"])


class PHECipher:
    def __init__(self, kind) -> None:
        self.kind = kind

    def keygen(self, **kwargs):
        from fate.arch.tensor import phe_keygen

        return phe_keygen(self.device, **kwargs)
