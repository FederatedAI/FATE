from enum import Enum
from typing import Tuple

from fate.interface import CipherKit as CipherKitInterface
from fate.interface import PHECipher as PHECipherInterface

from ..tensor._tensor import PHEDecryptor, PHEEncryptor
from ..unify import Backend, Device


class CipherKit(CipherKitInterface):
    def __init__(self, backend: Backend, device: Device) -> None:
        self.backend = backend
        self.device = device

    @property
    def phe(self):
        return PHECipher(self.backend, self.device)


class PHEKind(Enum):
    AUTO = "auto"
    PAILLIER = "Paillier"
    RUST_PAILLIER = "rust_paillier"
    INTEL_PAILLIER = "intel_paillier"


class PHECipher(PHECipherInterface):
    def __init__(self, backend: Backend, device: Device) -> None:
        self.backend = backend
        self.device = device

    def keygen(
        self, kind: PHEKind = PHEKind.AUTO, options={}
    ) -> Tuple["PHEEncryptor", "PHEDecryptor"]:
        if kind == PHEKind.AUTO or PHEKind.PAILLIER:

            if self.backend == Backend.LOCAL:
                from ..tensor.impl.tensor.multithread_cpu_tensor import (
                    PaillierPHECipherLocal,
                )

                key_length = options.get("key_length", 1024)
                encryptor, decryptor = PaillierPHECipherLocal().keygen(
                    key_length=key_length
                )
                return PHEEncryptor(encryptor), PHEDecryptor(decryptor)

            if self.backend in {Backend.STANDALONE, Backend.SPARK, Backend.EGGROLL}:
                from ..tensor.impl.tensor.distributed import (
                    PaillierPHECipherDistributed,
                )

                key_length = options.get("key_length", 1024)
                encryptor, decryptor = PaillierPHECipherDistributed().keygen(
                    key_length=key_length
                )
                return PHEEncryptor(encryptor), PHEDecryptor(decryptor)

        raise NotImplementedError(
            f"keygen for kind<{kind}>-distributed<{self.backend}>-device<{self.device}> is not implemented"
        )
