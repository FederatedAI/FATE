from enum import Enum
from typing import Protocol, Tuple

from ._tensor import Tensor


class PHEKind(Enum):
    AUTO = "auto"
    PAILLIER = "Paillier"
    RUST_PAILLIER = "rust_paillier"
    INTEL_PAILLIER = "intel_paillier"


class PHECipher(Protocol):
    def keygen(
        self, kind: PHEKind = PHEKind.AUTO, options={}
    ) -> Tuple["PHEEncryptor", "PHEDecryptor"]:
        ...


class CipherKit(Protocol):
    phe: PHECipher


class PHEEncryptor(Protocol):
    def encrypt(self, tensor) -> Tensor:
        ...


class PHEDecryptor(Protocol):
    def decrypt(self, tensor) -> Tensor:
        ...
