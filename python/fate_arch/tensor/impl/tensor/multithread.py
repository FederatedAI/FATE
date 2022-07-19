import typing
from typing import Union
import operator
import torch

from ...abc.tensor import (
    FPTensorProtocol,
    PHECipherABC,
    PHEDecryptorABC,
    PHEEncryptorABC,
    PHETensorABC,
)

FPTensorLocal = torch.Tensor
Numeric = typing.Union[int, float]
TYPEFP = typing.Union[Numeric, "FPTensorLocal"]
TYPECT = typing.Union[TYPEFP, "PHETensorLocal"]


class PHETensorLocal(PHETensorABC):
    def __init__(self, block) -> None:
        """ """
        self._block = block
        self._is_transpose = False

    def __add__(self, other: TYPECT) -> "PHETensorLocal":
        if isinstance(other, PHETensorLocal):
            other = other._block
        return _phe_binary_op(self._block, other, operator.add, PHE_OP_TYPES)

    def __radd__(self, other: TYPECT) -> "PHETensorLocal":
        if isinstance(other, PHETensorLocal):
            other = other._block
        return _phe_binary_op(other, self._block, operator.add, PHE_OP_TYPES)

    def __sub__(self, other: TYPECT) -> "PHETensorLocal":
        if isinstance(other, PHETensorLocal):
            other = other._block
        return _phe_binary_op(self._block, other, operator.sub, PHE_OP_TYPES)

    def __rsub__(self, other: TYPECT) -> "PHETensorLocal":
        if isinstance(other, PHETensorLocal):
            other = other._block
        return _phe_binary_op(other, self._block, operator.sub, PHE_OP_TYPES)

    def __mul__(self, other: TYPEFP) -> "PHETensorLocal":
        return _phe_binary_op(self._block, other, operator.mul, PHE_OP_PLAIN_TYPES)

    def __rmul__(self, other: TYPEFP) -> "PHETensorLocal":
        return _phe_binary_op(other, self._block, operator.mul, PHE_OP_PLAIN_TYPES)

    def __matmul__(self, other: FPTensorLocal) -> "PHETensorLocal":
        if isinstance(other, FPTensorLocal):
            return PHETensorLocal(operator.matmul(self._block, other))
        return NotImplemented

    def __rmatmul__(self, other: FPTensorLocal) -> "PHETensorLocal":
        if isinstance(other, FPTensorLocal):
            return PHETensorLocal(operator.matmul(other, self._block))
        return NotImplemented

    def T(self) -> "PHETensorLocal":
        transposed = PHETensorLocal(self._block)
        transposed._is_transpose = not self._is_transpose
        return transposed

    def serialize(self):
        # todo: impl me
        ...


class PaillierPHEEncryptorLocal(PHEEncryptorABC):
    def __init__(self, block_encryptor) -> None:
        self._block_encryptor = block_encryptor

    def encrypt(self, tensor: FPTensorLocal) -> PHETensorLocal:
        return PHETensorLocal(self._block_encryptor.encrypt(tensor))


class PaillierPHEDecryptorLocal(PHEDecryptorABC):
    def __init__(self, block_decryptor) -> None:
        self._block_decryptor = block_decryptor

    def decrypt(self, tensor: PHETensorLocal) -> FPTensorLocal:
        return self._block_decryptor.decrypt(tensor._block)


class PaillierPHECipherLocal(PHECipherABC):
    @classmethod
    def keygen(
        cls, **kwargs
    ) -> typing.Tuple[PaillierPHEEncryptorLocal, PaillierPHEDecryptorLocal]:
        from ..blocks.rust_paillier_block import BlockPaillierCipher

        block_encrytor, block_decryptor = BlockPaillierCipher.keygen(**kwargs)
        return (
            PaillierPHEEncryptorLocal(block_encrytor),
            PaillierPHEDecryptorLocal(block_decryptor),
        )

def _phe_binary_op(self, other, func, types):
    if type(other) not in types:
        return NotImplemented
    elif isinstance(other, (PHETensorLocal, FPTensorLocal)):
        return PHETensorLocal(func(self, other))
    elif isinstance(other, (int, float)):
        return PHETensorLocal(func(self, other))
    else:
        return NotImplemented


PHE_OP_PLAIN_TYPES = {int, float, FPTensorLocal, PHETensorLocal}
PHE_OP_TYPES = {int, float, FPTensorLocal, PHETensorLocal}
FP_OP_TYPES = {int, float, FPTensorLocal}
