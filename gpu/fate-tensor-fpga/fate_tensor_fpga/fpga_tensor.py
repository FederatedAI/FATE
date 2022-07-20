#
#  Copyright 2022 The FATE Authors. All Rights Reserved.
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

import typing

import numpy as np

from .fpga_engine import (
    PaillierEncryptedStorage,
    TensorShapeStorage,
    pi_add,
    te_p2c,
    fp_encode,
    pi_encrypt,
    pi_mul,
    pi_matmul,
    pi_rmatmul,
    pi_sum,
    pi_p2c_pub_key,
    pi_decrypt,
    pi_p2c_priv_key,
    te_c2p,
)
from .secureprotol.fate_paillier import (
    PaillierPublicKey,
    PaillierPrivateKey,
    PaillierKeypair,
)


class Cipherblock:
    def __init__(
            self,
            store: PaillierEncryptedStorage,
            shape: TensorShapeStorage,
            pk: "PK"):
        self.store = store
        self.shape = shape
        self.pk = pk

    def get_shape(self):
        return self.shape.to_tuple()

    def get_size(self):
        return self.shape.size()

    @staticmethod
    def gen_shape(other):
        return TensorShapeStorage().from_tuple(other.shape)

    def _add_plaintext(self, other) -> "Cipherblock":
        fp_store = fp_encode(
            te_p2c(other),
            self.pk.pub_key.n,
            self.pk.pub_key.max_int)
        pi_store = pi_encrypt(self.pk.cpu_pub_key, fp_store)
        res_store, res_shape = pi_add(
            self.pk.cpu_pub_key, self.store, pi_store, self.shape, self.gen_shape(other))
        return Cipherblock(res_store, res_shape, self.pk)

    def _mul_plaintext(self, other) -> "Cipherblock":
        fp_store = fp_encode(
            te_p2c(other),
            self.pk.pub_key.n,
            self.pk.pub_key.max_int)
        res_store, res_shape = pi_mul(
            self.pk.cpu_pub_key, self.store, fp_store, self.shape, self.gen_shape(other))
        return Cipherblock(res_store, res_shape, self.pk)

    def _matmul_plaintext(self, other) -> "Cipherblock":
        fp_store = fp_encode(
            te_p2c(other),
            self.pk.pub_key.n,
            self.pk.pub_key.max_int)
        res_store, res_shape = pi_matmul(
            self.pk.cpu_pub_key, self.store, fp_store, self.shape, self.gen_shape(other))
        return Cipherblock(res_store, res_shape, self.pk)

    def _rmatmul_plaintext(self, other) -> "Cipherblock":
        fp_store = fp_encode(
            te_p2c(other),
            self.pk.pub_key.n,
            self.pk.pub_key.max_int)
        res_store, res_shape = pi_rmatmul(
            self.pk.cpu_pub_key, fp_store, self.store, self.gen_shape(other), self.shape)
        return Cipherblock(res_store, res_shape, self.pk)

    def add_cipherblock(self, other: "Cipherblock") -> "Cipherblock":
        res_store, res_shape = pi_add(
            self.pk.cpu_pub_key, self.store, other.store, self.shape, other.shape)
        return Cipherblock(res_store, res_shape, self.pk)

    def add_plaintext_f64(self, other) -> "Cipherblock":
        return self._add_plaintext(other)

    def add_plaintext_f32(self, other) -> "Cipherblock":
        return self._add_plaintext(other)

    def add_plaintext_i64(self, other) -> "Cipherblock":
        return self._add_plaintext(other)

    def add_plaintext_i32(self, other) -> "Cipherblock":
        return self._add_plaintext(other)

    def add_plaintext_scalar_f64(
        self, other: typing.Union[float, np.float64]
    ) -> "Cipherblock":
        other_array = np.asarray([other], dtype=np.float64)
        return self._add_plaintext(other_array)

    def add_plaintext_scalar_f32(
        self, other: typing.Union[float, np.float32]
    ) -> "Cipherblock":
        other_array = np.asarray([other], dtype=np.float32)
        return self._add_plaintext(other_array)

    def add_plaintext_scalar_i64(
        self, other: typing.Union[int, np.int64]
    ) -> "Cipherblock":
        other_array = np.asarray([other], dtype=np.int64)
        return self._add_plaintext(other_array)

    def add_plaintext_scalar_i32(
        self, other: typing.Union[int, np.int32]
    ) -> "Cipherblock":
        other_array = np.asarray([other], dtype=np.int32)
        return self._add_plaintext(other_array)

    def sub_cipherblock(self, other: "Cipherblock") -> "Cipherblock":
        return self.add_cipherblock(other.mul_plaintext_scalar_i32(-1))

    def sub_plaintext_f64(self, other) -> "Cipherblock":
        return self.add_plaintext_f64(other * -1)

    def sub_plaintext_f32(self, other) -> "Cipherblock":
        return self.add_plaintext_f32(other * -1)

    def sub_plaintext_i64(self, other) -> "Cipherblock":
        return self.add_plaintext_i64(other * -1)

    def sub_plaintext_i32(self, other) -> "Cipherblock":
        return self.add_plaintext_i32(other * -1)

    def sub_plaintext_scalar_f64(
        self, other: typing.Union[float, np.float64]
    ) -> "Cipherblock":
        return self.add_plaintext_scalar_f64(other * -1)

    def sub_plaintext_scalar_f32(
        self, other: typing.Union[float, np.float32]
    ) -> "Cipherblock":
        return self.add_plaintext_scalar_f32(other * -1)

    def sub_plaintext_scalar_i64(
        self, other: typing.Union[int, np.int64]
    ) -> "Cipherblock":
        return self.add_plaintext_scalar_i64(other * -1)

    def sub_plaintext_scalar_i32(
        self, other: typing.Union[int, np.int32]
    ) -> "Cipherblock":
        return self.add_plaintext_scalar_i32(other * -1)

    def mul_plaintext_f64(self, other) -> "Cipherblock":
        return self._mul_plaintext(other)

    def mul_plaintext_f32(self, other) -> "Cipherblock":
        return self._mul_plaintext(other)

    def mul_plaintext_i64(self, other) -> "Cipherblock":
        return self._mul_plaintext(other)

    def mul_plaintext_i32(self, other) -> "Cipherblock":
        return self._mul_plaintext(other)

    def mul_plaintext_scalar_f64(
        self, other: typing.Union[float, np.float64]
    ) -> "Cipherblock":
        other_array = np.asarray([other], dtype=np.float64)
        return self._mul_plaintext(other_array)

    def mul_plaintext_scalar_f32(
        self, other: typing.Union[float, np.float32]
    ) -> "Cipherblock":
        other_array = np.asarray([other], dtype=np.float32)
        return self._mul_plaintext(other_array)

    def mul_plaintext_scalar_i64(
        self, other: typing.Union[int, np.int64]
    ) -> "Cipherblock":
        other_array = np.asarray([other], dtype=np.int64)
        return self._mul_plaintext(other_array)

    def mul_plaintext_scalar_i32(
        self, other: typing.Union[int, np.int32]
    ) -> "Cipherblock":
        other_array = np.asarray([other], dtype=np.int32)
        return self._mul_plaintext(other_array)

    def matmul_plaintext_ix2_f64(self, other) -> "Cipherblock":
        return self._matmul_plaintext(other)

    def matmul_plaintext_ix2_f32(self, other) -> "Cipherblock":
        return self._matmul_plaintext(other)

    def matmul_plaintext_ix2_i64(self, other) -> "Cipherblock":
        return self._matmul_plaintext(other)

    def matmul_plaintext_ix2_i32(self, other) -> "Cipherblock":
        return self._matmul_plaintext(other)

    def matmul_plaintext_ix1_f64(self, other) -> "Cipherblock":
        return self._matmul_plaintext(other)

    def matmul_plaintext_ix1_f32(self, other) -> "Cipherblock":
        return self._matmul_plaintext(other)

    def matmul_plaintext_ix1_i64(self, other) -> "Cipherblock":
        return self._matmul_plaintext(other)

    def matmul_plaintext_ix1_i32(self, other) -> "Cipherblock":
        return self._matmul_plaintext(other)

    def rmatmul_plaintext_ix2_f64(self, other) -> "Cipherblock":
        return self._rmatmul_plaintext(other)

    def rmatmul_plaintext_ix2_f32(self, other) -> "Cipherblock":
        return self._rmatmul_plaintext(other)

    def rmatmul_plaintext_ix2_i64(self, other) -> "Cipherblock":
        return self._rmatmul_plaintext(other)

    def rmatmul_plaintext_ix2_i32(self, other) -> "Cipherblock":
        return self._rmatmul_plaintext(other)

    def rmatmul_plaintext_ix1_f64(self, other) -> "Cipherblock":
        return self._rmatmul_plaintext(other)

    def rmatmul_plaintext_ix1_f32(self, other) -> "Cipherblock":
        return self._rmatmul_plaintext(other)

    def rmatmul_plaintext_ix1_i64(self, other) -> "Cipherblock":
        return self._rmatmul_plaintext(other)

    def rmatmul_plaintext_ix1_i32(self, other) -> "Cipherblock":
        return self._rmatmul_plaintext(other)

    def sum(self) -> "Cipherblock":
        res_store, res_shape = pi_sum(
            self.pk.cpu_pub_key, self.store, self.shape, axis=None
        )
        return Cipherblock(res_store, res_shape, self.pk)

    def sum_axis(self, axis=None):
        res_store, res_shape = pi_sum(
            self.pk.cpu_pub_key, self.store, self.shape, axis)
        return Cipherblock(res_store, res_shape, self.pk)

    def mean(self) -> "Cipherblock":
        return self.sum().mul_plaintext_scalar_f64(float(1 / self.get_size()))

    """parallel"""

    def add_cipherblock_par(self, other: "Cipherblock") -> "Cipherblock":
        return self.add_cipherblock(other)

    def add_plaintext_f64_par(self, other) -> "Cipherblock":
        return self.add_plaintext_f64(other)

    def add_plaintext_f32_par(self, other) -> "Cipherblock":
        return self.add_plaintext_f32(other)

    def add_plaintext_i64_par(self, other) -> "Cipherblock":
        return self.add_plaintext_i64(other)

    def add_plaintext_scalar_f64_par(
        self, other: typing.Union[float, np.float64]
    ) -> "Cipherblock":
        return self.add_plaintext_scalar_f64(other)

    def add_plaintext_scalar_f32_par(
        self, other: typing.Union[float, np.float32]
    ) -> "Cipherblock":
        return self.add_plaintext_scalar_f32(other)

    def add_plaintext_scalar_i64_par(
        self, other: typing.Union[int, np.int64]
    ) -> "Cipherblock":
        return self.add_plaintext_scalar_i64(other)

    def add_plaintext_scalar_i32_par(
        self, other: typing.Union[int, np.int32]
    ) -> "Cipherblock":
        return self.add_plaintext_scalar_i32(other)

    def add_plaintext_i32_par(self, other) -> "Cipherblock":
        return self.add_plaintext_i32(other)

    def sub_cipherblock_par(self, other: "Cipherblock") -> "Cipherblock":
        return self.sub_cipherblock(other)

    def sub_plaintext_f64_par(self, other) -> "Cipherblock":
        return self.sub_plaintext_f64(other)

    def sub_plaintext_f32_par(self, other) -> "Cipherblock":
        return self.sub_plaintext_f32(other)

    def sub_plaintext_i64_par(self, other) -> "Cipherblock":
        return self.sub_plaintext_i64(other)

    def sub_plaintext_i32_par(self, other) -> "Cipherblock":
        return self.sub_plaintext_i32(other)

    def sub_plaintext_scalar_f64_par(
        self, other: typing.Union[float, np.float64]
    ) -> "Cipherblock":
        return self.sub_plaintext_scalar_f64(other)

    def sub_plaintext_scalar_f32_par(
        self, other: typing.Union[float, np.float32]
    ) -> "Cipherblock":
        return self.sub_plaintext_scalar_f32(other)

    def sub_plaintext_scalar_i64_par(
        self, other: typing.Union[int, np.int64]
    ) -> "Cipherblock":
        return self.sub_plaintext_scalar_i64(other)

    def sub_plaintext_scalar_i32_par(
        self, other: typing.Union[int, np.int32]
    ) -> "Cipherblock":
        return self.sub_plaintext_scalar_i32(other)

    def mul_plaintext_f64_par(self, other) -> "Cipherblock":
        return self.mul_plaintext_f64(other)

    def mul_plaintext_f32_par(self, other) -> "Cipherblock":
        return self.mul_plaintext_f32(other)

    def mul_plaintext_i64_par(self, other) -> "Cipherblock":
        return self.mul_plaintext_i64(other)

    def mul_plaintext_i32_par(self, other) -> "Cipherblock":
        return self.mul_plaintext_i32(other)

    def mul_plaintext_scalar_f64_par(
        self, other: typing.Union[float, np.float64]
    ) -> "Cipherblock":
        return self.mul_plaintext_scalar_f64(other)

    def mul_plaintext_scalar_f32_par(
        self, other: typing.Union[float, np.float32]
    ) -> "Cipherblock":
        return self.mul_plaintext_scalar_f32(other)

    def mul_plaintext_scalar_i64_par(
        self, other: typing.Union[int, np.int64]
    ) -> "Cipherblock":
        return self.mul_plaintext_scalar_i64(other)

    def mul_plaintext_scalar_i32_par(
        self, other: typing.Union[int, np.int32]
    ) -> "Cipherblock":
        return self.mul_plaintext_scalar_i32(other)

    def matmul_plaintext_ix2_f64_par(self, other) -> "Cipherblock":
        return self.matmul_plaintext_ix2_f64(other)

    def matmul_plaintext_ix2_f32_par(self, other) -> "Cipherblock":
        return self.matmul_plaintext_ix2_f32(other)

    def matmul_plaintext_ix2_i64_par(self, other) -> "Cipherblock":
        return self.matmul_plaintext_ix2_i64(other)

    def matmul_plaintext_ix2_i32_par(self, other) -> "Cipherblock":
        return self.matmul_plaintext_ix2_i32(other)

    def matmul_plaintext_ix1_f64_par(self, other) -> "Cipherblock":
        return self.matmul_plaintext_ix1_f64(other)

    def matmul_plaintext_ix1_f32_par(self, other) -> "Cipherblock":
        return self.matmul_plaintext_ix1_f32(other)

    def matmul_plaintext_ix1_i64_par(self, other) -> "Cipherblock":
        return self.matmul_plaintext_ix1_i64(other)

    def matmul_plaintext_ix1_i32_par(self, other) -> "Cipherblock":
        return self.matmul_plaintext_ix1_i32(other)

    def rmatmul_plaintext_ix2_f64_par(self, other) -> "Cipherblock":
        return self.rmatmul_plaintext_ix2_f64(other)

    def rmatmul_plaintext_ix2_f32_par(self, other) -> "Cipherblock":
        return self.rmatmul_plaintext_ix2_f32(other)

    def rmatmul_plaintext_ix2_i64_par(self, other) -> "Cipherblock":
        return self.rmatmul_plaintext_ix2_i64(other)

    def rmatmul_plaintext_ix2_i32_par(self, other) -> "Cipherblock":
        return self.rmatmul_plaintext_ix2_i32(other)

    def rmatmul_plaintext_ix1_f64_par(self, other) -> "Cipherblock":
        return self.rmatmul_plaintext_ix1_f64(other)

    def rmatmul_plaintext_ix1_f32_par(self, other) -> "Cipherblock":
        return self.rmatmul_plaintext_ix1_f32(other)

    def rmatmul_plaintext_ix1_i64_par(self, other) -> "Cipherblock":
        return self.rmatmul_plaintext_ix1_i64(other)

    def rmatmul_plaintext_ix1_i32_par(self, other) -> "Cipherblock":
        return self.rmatmul_plaintext_ix1_i32(other)

    def sum_par(self) -> "Cipherblock":
        return self.sum()

    def mean_par(self) -> "Cipherblock":
        return self.mean()


class PK:
    def __init__(self, pub_key: PaillierPublicKey):
        self.pub_key = pub_key
        self.cpu_pub_key = pi_p2c_pub_key(None, self.pub_key)

    def _encrypt(self, a) -> Cipherblock:
        shape = TensorShapeStorage().from_tuple(a.shape)
        fp_store = fp_encode(te_p2c(a), self.pub_key.n, self.pub_key.max_int)
        pi_store = pi_encrypt(self.cpu_pub_key, fp_store)
        return Cipherblock(pi_store, shape, self)

    def encrypt_f64(self, a) -> Cipherblock:
        return self._encrypt(a)

    def encrypt_f32(self, a) -> Cipherblock:
        return self._encrypt(a)

    def encrypt_i64(self, a) -> Cipherblock:
        return self._encrypt(a)

    def encrypt_i32(self, a) -> Cipherblock:
        return self._encrypt(a)

    def encrypt_f64_par(self, a) -> Cipherblock:
        return self.encrypt_f64(a)

    def encrypt_f32_par(self, a) -> Cipherblock:
        return self.encrypt_f32(a)

    def encrypt_i64_par(self, a) -> Cipherblock:
        return self.encrypt_i64(a)

    def encrypt_i32_par(self, a) -> Cipherblock:
        return self.encrypt_i32(a)


class SK:
    def __init__(self, priv_key: PaillierPrivateKey, pk: PK):
        self.priv_key = priv_key
        self.cpu_priv_key = pi_p2c_priv_key(None, priv_key)
        self.pk = pk

    def _decrypt(self, a: Cipherblock):
        if a.store.vec_size == 0:
            return np.asarray([])
        te_res = pi_decrypt(a.pk.cpu_pub_key, self.cpu_priv_key, a.store)
        return te_c2p(te_res).reshape(a.get_shape())

    def decrypt_f64(self, a: Cipherblock):
        return self._decrypt(a).astype(np.float64)

    def decrypt_f32(self, a: Cipherblock):
        return self._decrypt(a).astype(np.float32)

    def decrypt_i64(self, a: Cipherblock):
        return self._decrypt(a).astype(np.int64)

    def decrypt_i32(self, a: Cipherblock):
        return self._decrypt(a).astype(np.int32)

    def decrypt_f64_par(self, a: Cipherblock):
        return self.decrypt_f64(a)

    def decrypt_f32_par(self, a: Cipherblock):
        return self.decrypt_f32(a)

    def decrypt_i64_par(self, a: Cipherblock):
        return self.decrypt_i64(a)

    def decrypt_i32_par(self, a: Cipherblock):
        return self.decrypt_i32(a)


def keygen(bit_size) -> typing.Tuple[PK, SK]:
    pub_key, priv_key = PaillierKeypair.generate_keypair(n_length=bit_size)
    pk = PK(pub_key)
    sk = SK(priv_key, pk)
    return pk, sk
