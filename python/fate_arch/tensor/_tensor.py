import typing
from enum import Enum
from typing import Union

from ._federation import FedKey, _Parties
from .abc.tensor import FPTensorABC, PHEDecryptorABC, PHEEncryptorABC, PHETensorABC


class FPTensor:
    @classmethod
    def zeors(cls, shape) -> "FPTensor":
        ...

    def __init__(self, tensor: FPTensorABC) -> None:
        self._tensor = tensor

    def __add__(self, other: Union["FPTensor", float, int]) -> "FPTensor":
        return self._binary_op(other, self._tensor.__add__)

    def __radd__(self, other: Union["FPTensor", float, int]) -> "FPTensor":
        return self._binary_op(other, self._tensor.__add__)

    def __sub__(self, other: Union["FPTensor", float, int]) -> "FPTensor":
        return self._binary_op(other, self._tensor.__sub__)

    def __rsub__(self, other: Union["FPTensor", float, int]) -> "FPTensor":
        return self._binary_op(other, self._tensor.__rsub__)

    def __mul__(self, other: Union["FPTensor", float, int]) -> "FPTensor":
        return self._binary_op(other, self._tensor.__mul__)

    def __rmul__(self, other: Union["FPTensor", float, int]) -> "FPTensor":
        return self._binary_op(other, self._tensor.__rmul__)

    def __matmul__(self, other: "FPTensor") -> "FPTensor":
        if isinstance(other, FPTensor):
            return FPTensor(self._tensor.__matmul__(other._tensor))
        else:
            return NotImplemented

    def __rmatmul__(self, other: "FPTensor") -> "FPTensor":
        if isinstance(other, FPTensor):
            return FPTensor(self._tensor.__rmatmul__(other._tensor))
        else:
            return NotImplemented

    def encrypted_phe(self, encryptor: "PHEEncryptor") -> "PHETensor":
        return encryptor.encrypt(self)

    @classmethod
    def pull_one(cls, source: _Parties, name: FedKey) -> "FPTensor":
        return name.pull_one(source)

    def push(self, target: _Parties, name: FedKey):
        return name.push(target, self)

    def _binary_op(self, other, func):
        if isinstance(other, FPTensor):
            return FPTensor(func(other._tensor))
        elif isinstance(other, (int, float)):
            return FPTensor(func(other))
        else:
            return NotImplemented


class PHETensor:
    def __init__(self, tensor: PHETensorABC) -> None:
        self._tensor = tensor

    def __add__(self, other: Union["PHETensor", FPTensor, int, float]) -> "PHETensor":
        return self._binary_op(other, self._tensor.__add__)

    def __radd__(self, other: Union["PHETensor", FPTensor, int, float]) -> "PHETensor":
        return self._binary_op(other, self._tensor.__radd__)

    def __sub__(self, other: Union["PHETensor", FPTensor, int, float]) -> "PHETensor":
        return self._binary_op(other, self._tensor.__sub__)

    def __rsub__(self, other: Union["PHETensor", FPTensor, int, float]) -> "PHETensor":
        return self._binary_op(other, self._tensor.__rsub__)

    def __mul__(self, other: Union[FPTensor, int, float]) -> "PHETensor":
        return self._binary_op(other, self._tensor.__mul__)

    def __rmul__(self, other: Union[FPTensor, int, float]) -> "PHETensor":
        return self._binary_op(other, self._tensor.__rmul__)

    def __matmul__(self, other: FPTensor) -> "PHETensor":
        if isinstance(other, FPTensor):
            return PHETensor(self._tensor.__matmul__(other._tensor))
        else:
            return NotImplemented

    def __rmatmul__(self, other: FPTensor) -> "PHETensor":
        if isinstance(other, FPTensor):
            return PHETensor(self._tensor.__rmatmul__(other._tensor))
        else:
            return NotImplemented

    def T(self) -> "PHETensor":
        return PHETensor(self._tensor.T())

    def decrypt_phe(self, decryptor: "PHEDecryptor") -> FPTensor:
        return decryptor.decrypt(self)

    @classmethod
    def pull_one(cls, source: _Parties, name: FedKey) -> "PHETensor":
        return name.pull_one(source)

    @classmethod
    def pull(cls, source: _Parties, name: FedKey) -> typing.List["PHETensor"]:
        return name.pull(source)

    def push(self, target: _Parties, name: FedKey):
        return name.push(target, self)

    def _binary_op(self, other, func):
        if isinstance(other, (PHETensor, FPTensor)):
            return PHETensor(func(other._tensor))
        elif isinstance(other, (int, float)):
            return PHETensor(func(other))
        return NotImplemented


class PHEEncryptor:
    def __init__(self, encryptor: PHEEncryptorABC) -> None:
        self._encryptor = encryptor

    def encrypt(self, tensor: FPTensor) -> PHETensor:
        return PHETensor(self._encryptor.encrypt(tensor._tensor))

    @classmethod
    def pull_one(cls, source: _Parties, name: FedKey) -> "PHEEncryptor":
        return PHEEncryptor(name.pull_one(source))

    def push(self, target: _Parties, name: FedKey):
        return name.push(target, self._encryptor)


class PHEDecryptor:
    def __init__(self, decryptor: PHEDecryptorABC) -> None:
        self._decryptor = decryptor

    def decrypt(self, tensor: PHETensor) -> FPTensor:
        return FPTensor(self._decryptor.decrypt(tensor._tensor))


class PHECipherKind(Enum):
    PAILLIER = 1


class PHECipher:
    def __init__(self, encryptor: PHEEncryptor, decryptor: PHEDecryptor) -> None:
        self._encryptor = encryptor
        self._decryptor = decryptor

    @classmethod
    def keygen(cls, kind: PHECipherKind, **kwargs) -> "PHECipher":
        if kind == PHECipherKind.PAILLIER:
            from .impl.tensor.distributed import PaillierPHECipherDistributed

            encryptor, decryptor = PaillierPHECipherDistributed().keygen()
            return PHECipher(PHEEncryptor(encryptor), PHEDecryptor(decryptor))
        else:
            raise NotImplementedError(
                f"{PHECipher.__name__} kind `{kind}` is not implemented"
            )

    @property
    def encryptor(self):
        return self._encryptor

    @property
    def decryptor(self):
        return self._decryptor
