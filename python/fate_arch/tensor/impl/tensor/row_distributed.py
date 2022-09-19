import typing
from typing import Union

from ...abc.tensor import (
    FPTensorProtocol,
    PHECipherABC,
    PHEDecryptorABC,
    PHEEncryptorABC,
    PHETensorABC,
)
from ..._federation import FederationDeserializer
from ..._tensor import Context, Party

Numeric = typing.Union[int, float]


class FPTensorDistributed(FPTensorProtocol):
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

    def __federation_hook__(self, ctx, key, parties):
        deserializer = FPTensorFederationDeserializer(key)
        # 1. remote deserializer with objs
        ctx._push(parties, key, deserializer)
        # 2. remote table
        ctx._push(parties, deserializer.table_key, self._blocks_table)


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
        self, other: FPTensorDistributed
    ) -> "PHETensorDistributed":
        # TODO: impl me
        ...

    def __rmatmul__(
        self, other: FPTensorDistributed
    ) -> "PHETensorDistributed":
        # TODO: impl me
        ...

    def T(self) -> "PHETensorDistributed":
        transposed = PHETensorDistributed(self._blocks_table)
        transposed._is_transpose = not self._is_transpose
        return transposed

    def serialize(self):
        return self._blocks_table

    def deserialize(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getstates__(self):
        return {"_is_transpose": self._is_transpose}

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

    def __federation_hook__(self, ctx, key, parties):
        deserializer = PHETensorFederationDeserializer(key, self._is_transpose)
        # 1. remote deserializer with objs
        ctx._push(parties, key, deserializer)
        # 2. remote table
        ctx._push(parties, deserializer.table_key, self._blocks_table)


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
        from ..blocks.cpu_paillier_block import BlockPaillierCipher

        block_encrytor, block_decryptor = BlockPaillierCipher.keygen(**kwargs)
        return (
            PaillierPHEEncryptorDistributed(block_encrytor),
            PaillierPHEDecryptorDistributed(block_decryptor),
        )


class PHETensorFederationDeserializer(FederationDeserializer):
    def __init__(self, key, is_transpose) -> None:
        self.table_key = self.make_frac_key(key, "table")
        self.is_transpose = is_transpose

    def do_deserialize(self, ctx: Context, party: Party) -> PHETensorDistributed:
        table = ctx._pull([party], self.table_key)[0]
        tensor = PHETensorDistributed(table)
        tensor._is_transpose = self.is_transpose
        return tensor


class FPTensorFederationDeserializer(FederationDeserializer):
    def __init__(self, key) -> None:
        self.table_key = self.make_frac_key(key, "table")

    def do_deserialize(self, ctx: Context, party: Party) -> FPTensorDistributed:
        table = ctx._pull([party], self.table_key)[0]
        tensor = FPTensorDistributed(table)
        return tensor
