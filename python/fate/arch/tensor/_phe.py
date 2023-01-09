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
from typing import Tuple

from fate.interface import PHECipher as PHECipherInterface

from ..unify import device


class PHEKind(Enum):
    AUTO = "auto"
    PAILLIER = "Paillier"
    RUST_PAILLIER = "rust_paillier"
    INTEL_PAILLIER = "intel_paillier"


class PHECipher(PHECipherInterface):
    def __init__(self, device: device) -> None:
        self.device = device

    def keygen(self, kind: PHEKind = PHEKind.AUTO, options={}) -> Tuple["PHEEncryptor", "PHEDecryptor"]:

        if kind == PHEKind.AUTO or PHEKind.PAILLIER:
            if self.device == device.CPU:
                from .storage.local.device.cpu.multithread_cpu_paillier_block import (
                    BlockPaillierCipher,
                )

                key_length = options.get("key_length", 1024)
                encryptor, decryptor = BlockPaillierCipher().keygen(key_length=key_length)
                return PHEEncryptor(encryptor), PHEDecryptor(decryptor)

        raise NotImplementedError(f"keygen for kind<{kind}>-device<{self.device}> is not implemented")


class PHEEncryptor:
    def __init__(self, storage_encryptor) -> None:
        self._encryptor = storage_encryptor

    def encrypt(self, tensor):
        from ..tensor import Tensor
        from .storage.local.device.cpu.paillier import _RustPaillierStorage
        from .types import DStorage, dtype

        if tensor.device == device.CPU:
            storage = tensor.storage
            if tensor.is_distributed:
                encrypted_storage = DStorage.elemwise_unary_op(
                    storage,
                    lambda s: _RustPaillierStorage(dtype.paillier, storage.shape, self._encryptor.encrypt(s.data)),
                    dtype.paillier,
                )
            else:
                encrypted_storage = _RustPaillierStorage(
                    dtype.paillier, storage.shape, self._encryptor.encrypt(storage.data)
                )
        else:
            raise NotImplementedError()
        return Tensor(encrypted_storage)


class PHEDecryptor:
    def __init__(self, storage_decryptor) -> None:
        self._decryptor = storage_decryptor

    def decrypt(self, tensor):
        from ..tensor import Tensor
        from .storage.local.device.cpu.plain import _TorchStorage
        from .types import DStorage, dtype

        storage = tensor.storage
        if isinstance(storage, DStorage):
            encrypted_storage = DStorage.elemwise_unary_op(
                storage,
                lambda s: _TorchStorage(dtype.paillier, storage.shape, self._decryptor.decrypt(s.data)),
                dtype.paillier,
            )
        else:
            encrypted_storage = _TorchStorage(dtype.float32, storage.shape, self._decryptor.decrypt(storage.data))
        return Tensor(encrypted_storage)
