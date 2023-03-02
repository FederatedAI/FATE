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
import pickle

import numpy as np
import torch


def _impl_ops(class_obj, method_name, ops):
    def func(self, other):
        cb = ops(self._cb, other, class_obj)
        if cb is NotImplemented:
            return NotImplemented
        else:
            return class_obj(cb)

    func.__name__ = method_name
    return func


def _impl_init():
    def __init__(self, cb):
        self._cb = cb

    return __init__


def _impl_encryptor_init():
    def __init__(self, pk):
        self._pk = pk

    return __init__


def _impl_decryptor_init():
    def __init__(self, sk):
        self._sk = sk

    return __init__


def _impl_encrypt(pheblock_cls, fpbloke_cls, encrypt_op):
    def encrypt(self, other) -> pheblock_cls:
        if isinstance(other, fpbloke_cls):
            return pheblock_cls(encrypt_op(self._pk, other.numpy()))

        raise NotImplementedError(f"type {other} not supported")

    return encrypt


def _impl_decrypt(pheblock_cls, fpbloke_cls, decrypt_op):
    def decrypt(self, other, dtype=np.float32) -> fpbloke_cls:
        if isinstance(other, pheblock_cls):
            return torch.from_numpy(decrypt_op(self._sk, other._cb, dtype))
        raise NotImplementedError(f"type {other} not supported")

    return decrypt


def _impl_serialize():
    def serialize(self) -> bytes:
        return pickle.dumps(self._cb)

    return serialize


def _impl_keygen(encrypt_cls, decrypt_cls, keygen_op):
    @classmethod
    def keygen(cls, key_length=1024):
        pk, sk = keygen_op(bit_size=key_length)
        return (encrypt_cls(pk), decrypt_cls(sk))

    return keygen


def _maybe_setattr(obj, name, value):
    if not hasattr(obj, name):
        setattr(obj, name, value)


def phe_keygen_metaclass(encrypt_cls, decrypt_cls, keygen_op):
    class PHEKeygenMetaclass(type):
        def __new__(cls, name, bases, dict):
            keygen_cls = super().__new__(cls, name, bases, dict)

            setattr(keygen_cls, "keygen", _impl_keygen(encrypt_cls, decrypt_cls, keygen_op))
            return keygen_cls

    return PHEKeygenMetaclass


def phe_decryptor_metaclass(pheblock_cls, fpblock_cls):
    class PHEDecryptorMetaclass(type):
        def __new__(cls, name, bases, dict):
            decryptor_cls = super().__new__(cls, name, bases, dict)

            setattr(decryptor_cls, "__init__", _impl_decryptor_init())
            setattr(
                decryptor_cls,
                "decrypt",
                _impl_decrypt(pheblock_cls, fpblock_cls, PHEDecryptorMetaclass._decrypt_numpy),
            )
            return decryptor_cls

        @staticmethod
        def _decrypt_numpy(sk, cb, dtype):
            if dtype == np.float64:
                return sk.decrypt_f64(cb)
            if dtype == np.float32:
                return sk.decrypt_f32(cb)
            if dtype == np.int64:
                return sk.decrypt_i64(cb)
            if dtype == np.int32:
                return sk.decrypt_i32(cb)
            raise NotImplementedError("dtype = {dtype}")

    return PHEDecryptorMetaclass


def phe_encryptor_metaclass(pheblock_cls, fpblock_cls):
    class PHEEncryptorMetaclass(type):
        def __new__(cls, name, bases, dict):
            encryptor_cls = super().__new__(cls, name, bases, dict)

            setattr(encryptor_cls, "__init__", _impl_encryptor_init())
            setattr(
                encryptor_cls,
                "encrypt",
                _impl_encrypt(pheblock_cls, fpblock_cls, PHEEncryptorMetaclass._encrypt_numpy),
            )
            return encryptor_cls

        @staticmethod
        def _encrypt_numpy(pk, other):
            if is_ndarray(other):
                if is_nd_float64(other):
                    return pk.encrypt_f64(other)
                if is_nd_float32(other):
                    return pk.encrypt_f32(other)
                if is_nd_int64(other):
                    return pk.encrypt_i64(other)
                if is_nd_int32(other):
                    return pk.encrypt_i32(other)
            raise NotImplementedError(f"type {other} {other.dtype} not supported")

    return PHEEncryptorMetaclass


class PHEBlockMetaclass(type):
    def __new__(cls, name, bases, dict):
        class_obj = super().__new__(cls, name, bases, dict)

        setattr(class_obj, "__init__", _impl_init())

        @property
        def shape(self):
            return self._cb.shape

        setattr(class_obj, "shape", shape)
        _maybe_setattr(class_obj, "serialize", _impl_serialize())
        for impl_name, ops in {
            "__add__": PHEBlockMetaclass._add,
            "__radd__": PHEBlockMetaclass._radd,
            "__sub__": PHEBlockMetaclass._sub,
            "__rsub__": PHEBlockMetaclass._rsub,
            "__mul__": PHEBlockMetaclass._mul,
            "__rmul__": PHEBlockMetaclass._rmul,
            "__matmul__": PHEBlockMetaclass._matmul,
            "__rmatmul__": PHEBlockMetaclass._rmatmul,
        }.items():
            _maybe_setattr(class_obj, impl_name, _impl_ops(class_obj, impl_name, ops))

        return class_obj

    @staticmethod
    def _rmatmul(cb, other, class_obj):
        if isinstance(other, torch.Tensor):
            other = other.numpy()
        if isinstance(other, np.ndarray):
            if len(other.shape) == 2:
                if is_nd_float64(other):
                    return cb.rmatmul_plaintext_ix2_f64(other)
                if is_nd_float32(other):
                    return cb.rmatmul_plaintext_ix2_f32(other)
                if is_nd_int64(other):
                    return cb.rmatmul_plaintext_ix2_i64(other)
                if is_nd_int32(other):
                    return cb.rmatmul_plaintext_ix2_i32(other)
            if len(other.shape) == 1:
                if is_nd_float64(other):
                    return cb.rmatmul_plaintext_ix1_f64(other)
                if is_nd_float32(other):
                    return cb.rmatmul_plaintext_ix1_f32(other)
                if is_nd_int64(other):
                    return cb.rmatmul_plaintext_ix1_i64(other)
                if is_nd_int32(other):
                    return cb.rmatmul_plaintext_ix1_i32(other)
        return NotImplemented

    @staticmethod
    def _matmul(cb, other, class_obj):
        if isinstance(other, torch.Tensor):
            other = other.numpy()
        if is_ndarray(other):
            if len(other.shape) == 2:
                if is_nd_float64(other):
                    return cb.matmul_plaintext_ix2_f64(other)
                if is_nd_float32(other):
                    return cb.matmul_plaintext_ix2_f32(other)
                if is_nd_int64(other):
                    return cb.matmul_plaintext_ix2_i64(other)
                if is_nd_int32(other):
                    return cb.matmul_plaintext_ix2_i32(other)
            if len(other.shape) == 1:
                if is_nd_float64(other):
                    return cb.matmul_plaintext_ix1_f64(other)
                if is_nd_float32(other):
                    return cb.matmul_plaintext_ix1_f32(other)
                if is_nd_int64(other):
                    return cb.matmul_plaintext_ix1_i64(other)
                if is_nd_int32(other):
                    return cb.matmul_plaintext_ix1_i32(other)
        return NotImplemented

    @staticmethod
    def _mul(cb, other, class_obj):
        if isinstance(other, torch.Tensor):
            other = other.numpy()
        if is_ndarray(other):
            if is_nd_float64(other):
                return cb.mul_plaintext_f64(other)
            if is_nd_float32(other):
                return cb.mul_plaintext_f32(other)
            if is_nd_int64(other):
                return cb.mul_plaintext_i64(other)
            if is_nd_int32(other):
                return cb.mul_plaintext_i32(other)
            raise NotImplemented
        if is_float(other):
            return cb.mul_plaintext_scalar_f64(other)
        if is_float32(other):
            return cb.mul_plaintext_scalar_f32(other)
        if is_int(other):
            return cb.mul_plaintext_scalar_i64(other)
        if is_int32(other):
            return cb.mul_plaintext_scalar_i32(other)
        return NotImplemented

    @staticmethod
    def _sub(cb, other, class_obj):
        if isinstance(other, torch.Tensor):
            other = other.numpy()
        if is_ndarray(other):
            if is_nd_float64(other):
                return cb.sub_plaintext_f64(other)
            if is_nd_float32(other):
                return cb.sub_plaintext_f32(other)
            if is_nd_int64(other):
                return cb.sub_plaintext_i64(other)
            if is_nd_int32(other):
                return cb.sub_plaintext_i32(other)
            return NotImplemented

        if isinstance(other, class_obj):
            return cb.sub_cipherblock(other._cb)
        if is_float(other):
            return cb.sub_plaintext_scalar_f64(other)
        if is_float32(other):
            return cb.sub_plaintext_scalar_f32(other)
        if is_int(other):
            return cb.sub_plaintext_scalar_i64(other)
        if is_int32(other):
            return cb.sub_plaintext_scalar_i32(other)

        return NotImplemented

    @staticmethod
    def _add(cb, other, class_obj):
        if isinstance(other, torch.Tensor):
            other = other.numpy()
        if is_ndarray(other):
            if is_nd_float64(other):
                return cb.add_plaintext_f64(other)
            if is_nd_float32(other):
                return cb.add_plaintext_f32(other)
            if is_nd_int64(other):
                return cb.add_plaintext_i64(other)
            if is_nd_int32(other):
                return cb.add_plaintext_i32(other)
            return NotImplemented

        if isinstance(other, class_obj):
            return cb.add_cipherblock(other._cb)
        if is_float(other):
            return cb.add_plaintext_scalar_f64(other)
        if is_float32(other):
            return cb.add_plaintext_scalar_f32(other)
        if is_int(other):
            return cb.add_plaintext_scalar_i64(other)
        if is_int32(other):
            return cb.add_plaintext_scalar_i32(other)

        return NotImplemented

    @staticmethod
    def _radd(cb, other, class_obj):
        return PHEBlockMetaclass._add(cb, other, class_obj)

    @staticmethod
    def _rsub(cb, other, class_obj):
        return PHEBlockMetaclass._add(PHEBlockMetaclass._mul(cb, -1, class_obj), other, class_obj)

    @staticmethod
    def _rmul(cb, other, class_obj):
        return PHEBlockMetaclass._mul(cb, other, class_obj)


def is_ndarray(v):
    return isinstance(v, np.ndarray)


def is_float(v):
    return isinstance(v, (float, np.float64))


def is_float32(v):
    return isinstance(v, np.float32)


def is_int(v):
    return isinstance(v, (int, np.int64))


def is_int32(v):
    return isinstance(v, np.int32)


def is_nd_float64(v):
    return v.dtype == np.float64


def is_nd_float32(v):
    return v.dtype == np.float32


def is_nd_int64(v):
    return v.dtype == np.int64


def is_nd_int32(v):
    return v.dtype == np.int32
