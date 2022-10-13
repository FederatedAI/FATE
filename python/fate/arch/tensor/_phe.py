from enum import Enum
from typing import Tuple

from fate.interface import PHECipher as PHECipherInterface

from ..unify import Backend, device


class PHEKind(Enum):
    AUTO = "auto"
    PAILLIER = "Paillier"
    RUST_PAILLIER = "rust_paillier"
    INTEL_PAILLIER = "intel_paillier"


class PHECipher(PHECipherInterface):
    def __init__(self, backend: Backend, device: device) -> None:
        self.backend = backend
        self.device = device

    def keygen(
        self, kind: PHEKind = PHEKind.AUTO, options={}
    ) -> Tuple["PHEEncryptor", "PHEDecryptor"]:

        if kind == PHEKind.AUTO or PHEKind.PAILLIER:
            if self.device == device.CPU:
                from ..tensor.impl.blocks.multithread_cpu_paillier_block import (
                    BlockPaillierCipher,
                )

                key_length = options.get("key_length", 1024)
                encryptor, decryptor = BlockPaillierCipher().keygen(
                    key_length=key_length
                )
                return PHEEncryptor(encryptor), PHEDecryptor(decryptor)

        raise NotImplementedError(
            f"keygen for kind<{kind}>-distributed<{self.backend}>-device<{self.device}> is not implemented"
        )


class PHEEncryptor:
    def __init__(self, storage_encryptor) -> None:
        self._encryptor = storage_encryptor

    def encrypt(self, tensor):
        from ..tensor._base import DStorage
        from ..tensor import Tensor
        from ..tensor.device.cpu import _CPUStorage
        from ..tensor._base import dtype

        storage = tensor.storage
        if isinstance(storage, DStorage):
            encrypted_storage = storage.elemwise_unary_op(
                lambda s: _CPUStorage(dtype.phe, self._encryptor.encrypt(s.data)),
                dtype.phe,
            )
        else:
            encrypted_storage = _CPUStorage(
                dtype.phe, self._encryptor.encrypt(storage.data)
            )
        return Tensor(encrypted_storage)


class PHEDecryptor:
    def __init__(self, storage_decryptor) -> None:
        self._decryptor = storage_decryptor

    def decrypt(self, tensor):
        from ..tensor._base import DStorage
        from ..tensor import Tensor
        from ..tensor.device.cpu import _CPUStorage
        from ..tensor._base import dtype

        storage = tensor.storage
        if isinstance(storage, DStorage):
            encrypted_storage = storage.elemwise_unary_op(
                lambda s: _CPUStorage(dtype.phe, self._decryptor.decrypt(s.data)),
                dtype.phe,
            )
        else:
            encrypted_storage = _CPUStorage(
                dtype.phe, self._decryptor.decrypt(storage.data)
            )
        return Tensor(encrypted_storage)
