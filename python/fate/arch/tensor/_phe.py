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
from enum import Enum

from ..unify import device
from ._ops import decrypt, encrypt


class PHEKind(Enum):
    AUTO = "auto"
    PAILLIER = "Paillier"
    RUST_PAILLIER = "rust_paillier"
    INTEL_PAILLIER = "intel_paillier"


class PHETensorEncryptor:
    def __init__(self, storage_encryptor) -> None:
        self._encryptor = storage_encryptor

    def encrypt(self, tensor):
        return encrypt(tensor, self._encryptor)


class PHETensorDecryptor:
    def __init__(self, storage_decryptor) -> None:
        self._decryptor = storage_decryptor

    def decrypt(self, tensor):
        return decrypt(tensor, self._decryptor)


def paillier_keygen(_device, key_length=1024):
    from ..storage._phe import paillier_keygen as paillier_keygen_storage

    encryptor, decryptor = paillier_keygen_storage(_device, key_length)
    return PHETensorEncryptor(encryptor), PHETensorDecryptor(decryptor)


def phe_keygen(kind: PHEKind, options={}, _device=device.CPU):
    if kind == PHEKind.AUTO or PHEKind.PAILLIER:
        if _device == device.CPU:
            return paillier_keygen(_device, **options)
    raise NotImplementedError(f"keygen for kind<{kind}>-device<{_device}> is not implemented")
