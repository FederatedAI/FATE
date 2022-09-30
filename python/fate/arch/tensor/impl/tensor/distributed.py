import typing
from typing import Union

import torch
from fate.interface import FederationDeserializer, FederationEngine, PartyMeta

from ....abc._computing import CTableABC
from ...abc.tensor import (
    FPTensorProtocol,
    PHECipherABC,
    PHEDecryptorABC,
    PHEEncryptorABC,
    PHETensorABC,
)

Numeric = typing.Union[int, float]


class Distributed:
    @property
    def blocks(self) -> CTableABC:
        ...

    def is_distributed(self):
        return True


class FPTensorDistributed(FPTensorProtocol, Distributed):
    """
    Demo of Distributed Fixed Presicion Tensor
    """

    def __init__(self, blocks_table, shape=None):
        """
        use table to store blocks in format (blockid, block)
        """
        self._blocks_table = blocks_table

        # assuming blocks are arranged vertically
        if shape is None:
            shapes = list(self._blocks_table.mapValues(lambda cb: cb.shape).collect())
            self.shape = (sum(s[0] for s in shapes), shapes[0][1])
        else:
            self.shape = shape

    @property
    def blocks(self):
        return self._blocks_table

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

    def collect(self):
        blocks = sorted(self._blocks_table.collect())
        return torch.cat([pair[1] for pair in blocks])

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

    def __matmul__(self, other: "PHETensorDistributed") -> "PHETensorDistributed":
        assert self.shape[1] == other.shape[0]
        # support one dimension only
        assert len(other.shape) == 1

        def func(cb):
            return cb @ other._blocks_table.collect()

        self._blocks_table.mapValues()

    def __rmatmul__(self, other: "PHETensorDistributed") -> "FPTensorDistributed":
        # todo: fix
        ...

    def __federation_hook__(self, ctx, key, parties):
        deserializer = FPTensorFederationDeserializer(key)
        # 1. remote deserializer with objs
        ctx._push(parties, key, deserializer)
        # 2. remote table
        ctx._push(parties, deserializer.table_key, self._blocks_table)


class PHETensorDistributed(PHETensorABC):
    def __init__(self, blocks_table, shape=None):
        """
        use table to store blocks in format (blockid, encrypted_block)
        """
        self._blocks_table = blocks_table
        self._is_transpose = False

        # assume block is verticel aranged
        if shape is None:
            shapes = list(self._blocks_table.mapValues(lambda cb: cb.shape).collect())
            self.shape = (sum(s[1][0] for s in shapes), shapes[0][1][1])
        else:
            self.shape = shape

    def collect(self):
        blocks = sorted(self._blocks_table.collect())
        return torch.cat([pair[1] for pair in blocks])

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

    def __matmul__(self, other: FPTensorDistributed) -> "PHETensorDistributed":
        # TODO: impl me
        ...

    def __rmatmul__(self, other: FPTensorDistributed) -> "PHETensorDistributed":
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
        self.table_key = f"distributed_phetensor_{key}__"
        self.is_transpose = is_transpose

    def __do_deserialize__(
        self,
        federation: FederationEngine,
        tag: str,
        party: PartyMeta,
    ) -> PHETensorDistributed:
        tensor = PHETensorDistributed(
            federation.pull(name=self.table_key, tag=tag, parties=[party])[0]
        )
        tensor._is_transpose = self.is_transpose
        return tensor


class FPTensorFederationDeserializer(FederationDeserializer):
    def __init__(self, key) -> None:
        self.table_key = f"distributed_tensor_{key}__"

    def __do_deserialize__(
        self,
        federation: FederationEngine,
        tag: str,
        party: PartyMeta,
    ) -> FPTensorDistributed:
        tensor = federation.pull(name=self.table_key, tag=tag, parties=[party])[0]
        return FPTensorDistributed(tensor)
