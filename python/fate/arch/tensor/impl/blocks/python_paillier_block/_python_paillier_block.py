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
#

import operator
import pickle
import typing

from torch import Tensor

from ....abc.block import (
    PHEBlockABC,
    PHEBlockCipherABC,
    PHEBlockDecryptorABC,
    PHEBlockEncryptorABC,
)
from ._fate_paillier import (
    PaillierEncryptedNumber,
    PaillierKeypair,
    PaillierPrivateKey,
    PaillierPublicKey,
)

# maybe need wrap?
FPBlock = Tensor

T = typing.TypeVar("T", bound=PaillierEncryptedNumber)

UnEncryptedNumeric = typing.Union[int, float, FPBlock]
EncryptedNumeric = typing.Union[PaillierEncryptedNumber, "PaillierBlock[T]"]
Numeric = typing.Union[UnEncryptedNumeric, EncryptedNumeric]


class PaillierBlock(typing.Generic[T], PHEBlockABC):
    """
    use list of list to mimic tensor
    """

    def __init__(self, inner: typing.List[typing.List[T]]) -> None:
        self._xsize = len(inner)
        self._ysize = len(inner[0])
        self.shape = (self._xsize, self._ysize)
        self._inner = inner

    @typing.overload
    def __getitem__(self, item: typing.Tuple[int, int]) -> T:
        ...

    @typing.overload
    def __getitem__(self, item: typing.Tuple[int, slice]) -> "PaillierBlock[T]":
        ...

    @typing.overload
    def __getitem__(self, item: typing.Tuple[slice, int]) -> "PaillierBlock[T]":
        ...

    @typing.overload
    def __getitem__(self, item: typing.Tuple[slice, slice]) -> "PaillierBlock[T]":
        ...

    @typing.overload
    def __getitem__(self, item: slice) -> "PaillierBlock[T]":
        ...

    def __getitem__(self, item):
        if isinstance(item, tuple) and len(item) == 2:
            xind, yind = item
            if isinstance(xind, int):
                return self._inner[xind][yind]
            elif isinstance(xind, slice):
                if isinstance(yind, slice):
                    return PaillierBlock([row[yind] for row in self._inner[xind]])
                if isinstance(yind, int):
                    return PaillierBlock([[row[yind]] for row in self._inner[xind]])
        elif isinstance(item, slice):
            return PaillierBlock(self._inner[item])
        return NotImplemented

    def _binary_op(
        self, other, tensor_types: typing.Tuple, scale_types: typing.Tuple, op
    ):
        if isinstance(other, tensor_types):
            assert self.shape == other.shape
            out = []
            for i in range(self._xsize):
                out.append([])
                for j in range(self._ysize):
                    out[i][j] = op(self[i, j], other[i, j])
            return PaillierBlock[T](out)
        elif isinstance(other, scale_types):
            out = []
            for i in range(self._xsize):
                out.append([])
                for j in range(self._ysize):
                    out[i][j] = op(self[i, j], other)
            return PaillierBlock[T](out)
        return NotImplemented

    def _binary_paillier_not_supported_op(self, other, op):
        return self._binary_op(other, (FPBlock,), (int, float), op)

    def _binary_paillier_supported_op(self, other, op):
        return self._binary_op(
            other, (PaillierBlock, FPBlock), (int, float, PaillierEncryptedNumber), op
        )

    def __add__(self, other: Numeric) -> "PaillierBlock":
        return self._binary_paillier_supported_op(other, operator.add)

    def __radd__(self, other: Numeric) -> "PaillierBlock":
        return self._binary_paillier_supported_op(other, operator.add)

    def __sub__(self, other: Numeric) -> "PaillierBlock":
        return self._binary_paillier_supported_op(other, operator.sub)

    def __rsub__(self, other: Numeric) -> "PaillierBlock":
        return self._binary_paillier_supported_op(other, lambda x, y: x.__rsub__(y))

    def __mul__(self, other: UnEncryptedNumeric) -> "PaillierBlock":
        return self._binary_paillier_not_supported_op(other, operator.mul)

    def __rmul__(self, other: UnEncryptedNumeric) -> "PaillierBlock":
        return self._binary_paillier_not_supported_op(other, lambda x, y: x.__rmul__(y))

    def __matmul__(self, other: FPBlock) -> "PaillierBlock":
        out = []
        if isinstance(other, FPBlock):
            assert self.shape[1] == other.shape[0]
            for i in range(self.shape[0]):
                out.append([])
                for j in range(other.shape[1]):
                    c = self[i, 0] * other[0, j]
                    for k in range(1, self.shape[1]):
                        c += self[i, k] * other[k, j]
                    out[i][j] = c
            return PaillierBlock(out)
        else:
            return NotImplemented

    def __rmatmul__(self, other: FPBlock) -> "PaillierBlock":
        out = []
        if isinstance(other, FPBlock):
            assert self.shape[0] == other.shape[1]
            for i in range(other.shape[0]):
                out.append([])
                for j in range(self.shape[1]):
                    c = other[i, 0] * self[0, j]
                    for k in range(1, other.shape[1]):
                        c += other[i, k] * self[k, j]
                    out[i][j] = c
            return PaillierBlock(out)
        else:
            return NotImplemented

    def serialize(self) -> bytes:
        return pickle.dumps(self._inner)

    def T(self) -> "PHEBlockABC":
        # todo: transpose could be lazy
        return PaillierBlock(
            [
                [self._inner[x][y] for y in range(self._ysize)]
                for x in range(self._xsize)
            ]
        )


class BlockPaillierEncryptor(PHEBlockEncryptorABC):
    def __init__(self, pubkey: PaillierPublicKey) -> None:
        self._pubkey = pubkey

    def encrypt(self, tensor: FPBlock) -> PaillierBlock:
        return PaillierBlock(
            [[self._pubkey.encrypt(x) for x in row] for row in tensor.tolist()],
        )


class BlockPaillierDecryptor(PHEBlockDecryptorABC):
    def __init__(self, prikey: PaillierPrivateKey) -> None:
        self._prikey = prikey

    def decrypt(self, tensor: PaillierBlock) -> FPBlock:
        return FPBlock(
            [[self._prikey.decrypt(x) for x in row] for row in tensor._inner]
        )


class BlockPaillierCipher(PHEBlockCipherABC):
    @classmethod
    def keygen(
        cls, n_length=1024, **kwargs
    ) -> typing.Tuple[BlockPaillierEncryptor, BlockPaillierDecryptor]:
        pubkey, prikey = PaillierKeypair.generate_keypair(n_length=n_length)
        return (BlockPaillierEncryptor(pubkey), BlockPaillierDecryptor(prikey))
