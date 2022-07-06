import typing
from typing import Union

from ...abc.tensor import (
    FPTensorABC,
    PHECipherABC,
    PHEDecryptorABC,
    PHEEncryptorABC,
    PHETensorABC,
)

Numeric = typing.Union[int, float]


class FPTensorDistributed(FPTensorABC):
    """
    Demo of Distributed Fixed Presicion Tensor
    """

    def __init__(self, blocks_table):
        """
        use table to store blocks in format (blockid, block)
        """
        self._blocks_table = blocks_table

    def _binary_op(self, other, func_name):
        if isinstance(other, FPTensorDistributed):
            return FPTensorDistributed(
                other._blocks_table.join(
                    self._blocks_table, lambda x, y: getattr(x, func_name)(y)
                )
            )
        elif isinstance(other, (int, float)):
            return FPTensorDistributed(
                self._blocks_table.mapValues(lambda x: getattr(x, func_name)(other))
            )
        return NotImplemented

    def __add__(
        self, other: Union["FPTensorDistributed", int, float]
    ) -> "FPTensorDistributed":
        return self._binary_op(other, "__add__")

    def __radd__(
        self, other: Union["FPTensorDistributed", int, float]
    ) -> "FPTensorDistributed":
        return self._binary_op(other, "__radd__")

    def __sub__(
        self, other: Union["FPTensorDistributed", int, float]
    ) -> "FPTensorDistributed":
        return self._binary_op(other, "__sub__")

    def __rsub__(
        self, other: Union["FPTensorDistributed", int, float]
    ) -> "FPTensorDistributed":
        return self._binary_op(other, "__rsub__")

    def __mul__(
        self, other: Union["FPTensorDistributed", int, float]
    ) -> "FPTensorDistributed":
        return self._binary_op(other, "__mul__")

    def __rmul__(
        self, other: Union["FPTensorDistributed", int, float]
    ) -> "FPTensorDistributed":
        return self._binary_op(other, "__rmul__")

    def __matmul__(self, other: "FPTensorDistributed") -> "FPTensorDistributed":
        # todo: fix
        ...

    def __rmatmul__(self, other: "FPTensorDistributed") -> "FPTensorDistributed":
        # todo: fix
        ...


class PHETensorDistributed(PHETensorABC):
    def __init__(self, blocks_table) -> None:
        """
        use table to store blocks in format (blockid, encrypted_block)
        """
        self._blocks_table = blocks_table
        self._is_transpose = False

    def __add__(
        self, other: Union["PHETensorDistributed", FPTensorDistributed, int, float]
    ) -> "PHETensorDistributed":
        return self._binary_op(other, "__add__")

    def __radd__(
        self, other: Union["PHETensorDistributed", FPTensorDistributed, int, float]
    ) -> "PHETensorDistributed":
        return self._binary_op(other, "__radd__")

    def __sub__(
        self, other: Union["PHETensorDistributed", FPTensorDistributed, int, float]
    ) -> "PHETensorDistributed":
        return self._binary_op(other, "__sub__")

    def __rsub__(
        self, other: Union["PHETensorDistributed", FPTensorDistributed, int, float]
    ) -> "PHETensorDistributed":
        return self._binary_op(other, "__rsub__")

    def __mul__(
        self, other: Union["PHETensorDistributed", FPTensorDistributed, int, float]
    ) -> "PHETensorDistributed":
        return self._binary_op_limited(other, "__mul__")

    def __rmul__(
        self, other: Union["PHETensorDistributed", FPTensorDistributed, int, float]
    ) -> "PHETensorDistributed":
        return self._binary_op_limited(other, "__rmul__")

    def __matmul__(
        self, other: Union["PHETensorDistributed", FPTensorDistributed, int, float]
    ) -> "PHETensorDistributed":
        # TODO: impl me
        ...

    def __rmatmul__(
        self, other: Union["PHETensorDistributed", FPTensorDistributed, int, float]
    ) -> "PHETensorDistributed":
        # TODO: impl me
        ...

    def T(self) -> "PHETensorDistributed":
        transposed = PHETensorDistributed(self._blocks_table)
        transposed._is_transpose = not self._is_transpose
        return transposed

    def serialize(self):
        # TODO: impl me
        ...

    def _binary_op(self, other, func_name):
        if isinstance(other, (FPTensorDistributed, PHETensorDistributed)):
            return PHETensorDistributed(
                self._blocks_table.join(
                    other._blocks_table, lambda x, y: getattr(x, func_name)(y)
                )
            )
        elif isinstance(other, (int, float)):
            return PHETensorDistributed(
                self._blocks_table.mapValues(lambda x: x.__add__(other))
            )

        return NotImplemented

    def _binary_op_limited(self, other, func_name):
        if isinstance(other, FPTensorDistributed):
            return PHETensorDistributed(
                self._blocks_table.join(
                    other._blocks_table, lambda x, y: getattr(x, func_name)(y)
                )
            )
        elif isinstance(other, (int, float)):
            return PHETensorDistributed(
                self._blocks_table.mapValues(lambda x: x.__add__(other))
            )
        return NotImplemented


class PaillierPHEEncryptorDistributed(PHEEncryptorABC):
    def __init__(self, block_encryptor) -> None:
        self._block_encryptor = block_encryptor

    def encrypt(self, tensor: FPTensorDistributed) -> PHETensorDistributed:
        return PHETensorDistributed(
            tensor._blocks_table.mapValues(lambda x: self._block_encryptor.encrypt(x))
        )


class PaillierPHEDecryptorDistributed(PHEDecryptorABC):
    def __init__(self, block_decryptor) -> None:
        self._block_decryptor = block_decryptor

    def decrypt(self, tensor: PHETensorDistributed) -> FPTensorDistributed:
        return FPTensorDistributed(
            tensor._blocks_table.mapValues(lambda x: self._block_decryptor.decrypt(x))
        )


class PaillierPHECipherDistributed(PHECipherABC):
    @classmethod
    def keygen(
        cls, **kwargs
    ) -> typing.Tuple[PaillierPHEEncryptorDistributed, PaillierPHEDecryptorDistributed]:
        from ..blocks.python_paillier_block import BlockPaillierCipher

        block_encrytor, block_decryptor = BlockPaillierCipher.keygen(**kwargs)
        return (
            PaillierPHEEncryptorDistributed(block_encrytor),
            PaillierPHEDecryptorDistributed(block_decryptor),
        )
