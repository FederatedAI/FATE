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

import pickle
import typing

import fate_tensor
import numpy as np
from fate_tensor import Cipherblock
import torch

from ....abc.block import (
    PHEBlockABC,
    PHEBlockCipherABC,
    PHEBlockDecryptorABC,
    PHEBlockEncryptorABC,
)

# maybe need wrap?
FPBlock = torch.Tensor

# TODO: move numpy related apis to rust side


class PaillierBlock(PHEBlockABC):
    def __init__(self, cb: Cipherblock) -> None:
        self._cb = cb

    def create(self, cb: Cipherblock):
        return PaillierBlock(cb)

    def __add__(self, other) -> "PaillierBlock":
        if isinstance(other, torch.Tensor):
            other = other.numpy()
        if isinstance(other, np.ndarray):
            if other.dtype == np.float64:
                return self.create(self._cb.add_plaintext_f64(other))
            if other.dtype == np.float32:
                return self.create(self._cb.add_plaintext_f32(other))
            if other.dtype == np.int64:
                return self.create(self._cb.add_plaintext_i64(other))
            if other.dtype == np.int64:
                return self.create(self._cb.add_plaintext_i32(other))
            raise NotImplemented(f"dtype {other.dtype} not supported")
        if isinstance(other, PaillierBlock):
            return self.create(self._cb.add_cipherblock(other._cb))
        if isinstance(other, (float, np.float64)):
            return self.create(self._cb.add_plaintext_scalar_f64(other))
        if isinstance(other, np.float32):
            return self.create(self._cb.add_plaintext_scalar_f32(other))
        if isinstance(other, (int, np.int64)):
            return self.create(self._cb.add_plaintext_scalar_i64(other))
        if isinstance(other, np.int32):
            return self.create(self._cb.add_plaintext_scalar_i32(other))
        raise NotImplemented(f"type {other} not supported")

    def __radd__(self, other) -> "PaillierBlock":
        return self.__add__(other)

    def __sub__(self, other) -> "PaillierBlock":
        if isinstance(other, torch.Tensor):
            other = other.numpy()
        if isinstance(other, np.ndarray):
            if other.dtype == np.float64:
                return self.create(self._cb.sub_plaintext_f64(other))
            if other.dtype == np.float32:
                return self.create(self._cb.sub_plaintext_f32(other))
            if other.dtype == np.int64:
                return self.create(self._cb.sub_plaintext_i64(other))
            if other.dtype == np.int32:
                return self.create(self._cb.sub_plaintext_i32(other))
            raise NotImplemented(f"dtype {other.dtype} not supported")
        if isinstance(other, PaillierBlock):
            return self.create(self._cb.sub_cipherblock(other._cb))
        if isinstance(other, (float, np.float64)):
            return self.create(self._cb.sub_plaintext_scalar_f64(other))
        if isinstance(other, np.float32):
            return self.create(self._cb.sub_plaintext_scalar_f32(other))
        if isinstance(other, (int, np.int64)):
            return self.create(self._cb.sub_plaintext_scalar_i64(other))
        if isinstance(other, np.int32):
            return self.create(self._cb.sub_plaintext_scalar_i32(other))
        raise NotImplemented(f"type {other} not supported")

    def __rsub__(self, other) -> "PaillierBlock":
        return self.__mul__(-1).__add__(other)

    def __mul__(self, other) -> "PaillierBlock":
        if isinstance(other, torch.Tensor):
            other = other.numpy()
        if isinstance(other, np.ndarray):
            if other.dtype == np.float64:
                return self.create(self._cb.mul_plaintext_f64(other))
            if other.dtype == np.float32:
                return self.create(self._cb.mul_plaintext_f32(other))
            if other.dtype == np.int64:
                return self.create(self._cb.mul_plaintext_i64(other))
            if other.dtype == np.int32:
                return self.create(self._cb.mul_plaintext_i32(other))
            raise NotImplemented(f"dtype {other.dtype} not supported")
        if isinstance(other, (float, np.float64)):
            return self.create(self._cb.mul_plaintext_scalar_f64(other))
        if isinstance(other, np.float32):
            return self.create(self._cb.mul_plaintext_scalar_f32(other))
        if isinstance(other, (int, np.int64)):
            return self.create(self._cb.mul_plaintext_scalar_i64(other))
        if isinstance(other, np.int32):
            return self.create(self._cb.mul_plaintext_scalar_i32(other))
        raise NotImplemented(f"type {other} not supported")

    def __rmul__(self, other) -> "PaillierBlock":
        return self.__mul__(other)

    def __matmul__(self, other) -> "PaillierBlock":
        if isinstance(other, torch.Tensor):
            other = other.numpy()
        if isinstance(other, np.ndarray):
            if len(other.shape) == 2:
                if other.dtype == np.float64:
                    return self.create(self._cb.matmul_plaintext_ix2_f64(other))
                if other.dtype == np.float32:
                    return self.create(self._cb.matmul_plaintext_ix2_f32(other))
                if other.dtype == np.int64:
                    return self.create(self._cb.matmul_plaintext_ix2_i64(other))
                if other.dtype == np.int32:
                    return self.create(self._cb.matmul_plaintext_ix2_i32(other))
            if len(other.shape) == 1:
                if other.dtype == np.float64:
                    return self.create(self._cb.matmul_plaintext_ix1_f64(other))
                if other.dtype == np.float32:
                    return self.create(self._cb.matmul_plaintext_ix1_f32(other))
                if other.dtype == np.int64:
                    return self.create(self._cb.matmul_plaintext_ix1_i64(other))
                if other.dtype == np.int32:
                    return self.create(self._cb.matmul_plaintext_ix1_i32(other))
        return NotImplemented

    def __rmatmul__(self, other) -> "PaillierBlock":
        if isinstance(other, torch.Tensor):
            other = other.numpy()
        if isinstance(other, np.ndarray):
            if len(other.shape) == 2:
                if other.dtype == np.float64:
                    return self.create(self._cb.rmatmul_plaintext_ix2_f64(other))
                if other.dtype == np.float32:
                    return self.create(self._cb.rmatmul_plaintext_ix2_f32(other))
                if other.dtype == np.int64:
                    return self.create(self._cb.rmatmul_plaintext_ix2_i64(other))
                if other.dtype == np.int32:
                    return self.create(self._cb.rmatmul_plaintext_ix2_i32(other))
            if len(other.shape) == 1:
                if other.dtype == np.float64:
                    return self.create(self._cb.rmatmul_plaintext_ix1_f64(other))
                if other.dtype == np.float32:
                    return self.create(self._cb.rmatmul_plaintext_ix1_f32(other))
                if other.dtype == np.int64:
                    return self.create(self._cb.rmatmul_plaintext_ix1_i64(other))
                if other.dtype == np.int32:
                    return self.create(self._cb.rmatmul_plaintext_ix1_i32(other))
        return NotImplemented

    def serialize(self) -> bytes:
        return pickle.dumps(self._cb)


class BlockPaillierEncryptor(PHEBlockEncryptorABC):
    def __init__(self, pk: fate_tensor.PK, multithread=False) -> None:
        self._pk = pk
        self._multithread = multithread

    def encrypt(self, other) -> PaillierBlock:
        if isinstance(other, FPBlock):
            return PaillierBlock(self._encrypt_numpy(other.numpy()))

        raise NotImplementedError(f"type {other} not supported")

    def _encrypt_numpy(self, other):
        if self._multithread:
            if isinstance(other, np.ndarray):
                if other.dtype == np.float64:
                    return self._pk.encrypt_f64_par(other)
                if other.dtype == np.float32:
                    return self._pk.encrypt_f32_par(other)
                if other.dtype == np.int64:
                    return self._pk.encrypt_i64_par(other)
                if other.dtype == np.int32:
                    return self._pk.encrypt_i32_par(other)
        else:
            if isinstance(other, np.ndarray):
                if other.dtype == np.float64:
                    return self._pk.encrypt_f64(other)
                if other.dtype == np.float32:
                    return self._pk.encrypt_f32(other)
                if other.dtype == np.int64:
                    return self._pk.encrypt_i64(other)
                if other.dtype == np.int32:
                    return self._pk.encrypt_i32(other)
        raise NotImplementedError(f"type {other} {other.dtype} not supported")


class BlockPaillierDecryptor(PHEBlockDecryptorABC):
    def __init__(self, sk: fate_tensor.SK, multithread=False) -> None:
        self._sk = sk
        self._multithread = multithread

    def decrypt(self, other: PaillierBlock, dtype=np.float64):
        return torch.from_numpy(self._decrypt_numpy(other._cb, dtype))

    def _decrypt_numpy(self, cb: Cipherblock, dtype=np.float64):
        if self._multithread:
            if dtype == np.float64:
                return self._sk.decrypt_f64_par(cb)
            if dtype == np.float32:
                return self._sk.decrypt_f32_par(cb)
            if dtype == np.int64:
                return self._sk.decrypt_i64_par(cb)
            if dtype == np.int32:
                return self._sk.decrypt_i32_par(cb)
        else:
            if dtype == np.float64:
                return self._sk.decrypt_f64(cb)
            if dtype == np.float32:
                return self._sk.decrypt_f32(cb)
            if dtype == np.int64:
                return self._sk.decrypt_i64(cb)
            if dtype == np.int32:
                return self._sk.decrypt_i32(cb)
        raise NotImplementedError("dtype = {dtype}")


class BlockPaillierCipher(PHEBlockCipherABC):
    @classmethod
    def keygen(
            cls, key_length=1024, multithread=False,
    ) -> typing.Tuple[BlockPaillierEncryptor, BlockPaillierDecryptor]:
        pubkey, prikey = fate_tensor.keygen(bit_size=key_length)
        return (BlockPaillierEncryptor(pubkey, multithread), BlockPaillierDecryptor(prikey, multithread))
