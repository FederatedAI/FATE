#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
import numpy as np
from federatedml.util import LOGGER
from fate_arch.session import computing_session
from fate_arch.abc import CTableABC


class PaillierTensor(object):
    def __init__(self, obj, partitions=1):
        if obj is None:
            raise ValueError("Cannot convert None to Paillier tensor")

        if isinstance(obj, (list, np.ndarray)):
            self._ori_data = obj
            self._partitions = partitions
            self._obj = computing_session.parallelize(obj,
                                                      include_key=False,
                                                      partition=partitions)
        elif isinstance(obj, CTableABC):
            self._ori_data = None
            self._partitions = obj.partitions
            self._obj = obj
        else:
            raise ValueError(f"Cannot convert obj to Paillier tensor, object type is {type(obj)}")

        LOGGER.debug("tensor's partition is {}".format(self._partitions))

    def __add__(self, other):
        if isinstance(other, PaillierTensor):
            return PaillierTensor(self._obj.join(other._obj, lambda v1, v2: v1 + v2))
        elif isinstance(other, CTableABC):
            return PaillierTensor(self._obj.join(other, lambda v1, v2: v1 + v2))
        elif isinstance(other, (np.ndarray, int, float)):
            return PaillierTensor(self._obj.mapValues(lambda v: v + other))
        else:
            raise ValueError(f"Unrecognized type {type(other)}, dose not support subtraction")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, PaillierTensor):
            return PaillierTensor(self._obj.join(other._obj, lambda v1, v2: v1 - v2))
        elif isinstance(other, CTableABC):
            return PaillierTensor(self._obj.join(other, lambda v1, v2: v1 - v2))
        elif isinstance(other, (np.ndarray, int, float)):
            return PaillierTensor(self._obj.mapValues(lambda v: v - other))
        else:
            raise ValueError(f"Unrecognized type {type(other)}, dose not support subtraction")

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return PaillierTensor(self._obj.mapValues(lambda val: val * other))
        elif isinstance(other, np.ndarray):
            return PaillierTensor(self._obj.mapValues(lambda val: np.matmul(val, other)))
        elif isinstance(other, CTableABC):
            other = PaillierTensor(other)
            return self.__mul__(other)
        elif isinstance(other, PaillierTensor):
            ret = self.numpy() * other.numpy()
            return PaillierTensor(ret, partitions=max(self.partitions, other.partitions))

    def matmul(self, other):
        if isinstance(other, np.ndarray):
            if len(other.shape) != 2:
                raise ValueError("Only Support 2-D multiplication in matmul op, "
                                 "if you want to do 3-D, use fast_multiply_3d")

        return self.fast_matmul_2d(other)

    def multiply(self, other):
        if isinstance(other, np.ndarray):
            if other.shape != self.shape:
                raise ValueError(f"operands could not be broadcast together with shapes {self.shape} {other.shape}")
            rhs = PaillierTensor(other)
            return PaillierTensor(self.multiply(rhs))
        elif isinstance(other, CTableABC):
            other = PaillierTensor(other)
            return self.multiply(other)
        elif isinstance(other, PaillierTensor):
            return PaillierTensor(self._obj.join(other._obj, lambda v1, v2: v1 * v2))
        else:
            raise ValueError(f"Not support type in multiply op {type(other)}")

    @property
    def T(self):
        if self._ori_data is None:
            self._ori_data = self.numpy()

        new_data = self._ori_data.T
        return PaillierTensor(new_data, self.partitions)

    @property
    def partitions(self):
        return self._partitions

    def get_obj(self):
        return self._obj

    @property
    def shape(self):
        if self._ori_data is not None:
            return self._ori_data.shape
        else:
            first_dim = self._obj.count()
            second_dim = self._obj.first()[1].shape

            return tuple([first_dim] + list(second_dim))

    def mean(self, axis=-1):
        if axis == -1:
            size = 1
            for shape in self._ori_data.shape:
                size *= shape

            if not size:
                raise ValueError("shape of data is zero, it should be positive")

            return self._obj.mapValues(lambda val: np.sum(val)).reduce(lambda val1, val2: val1 + val2) / size

        else:
            ret_obj = self._obj.mapValues(lambda val: np.mean(val, axis - 1))

            return PaillierTensor(ret_obj)

    def reduce_sum(self):
        return self._obj.reduce(lambda t1, t2: t1 + t2)

    def map_ndarray_product(self, other):
        if isinstance(other, np.ndarray):
            return PaillierTensor(self._obj.mapValues(lambda val: val * other))
        else:
            raise ValueError('only support numpy array')

    def numpy(self):
        if self._ori_data is not None:
            return self._ori_data

        arr = [None for i in range(self._obj.count())]

        for k, v in self._obj.collect():
            arr[k] = v

        self._ori_data = np.array(arr, dtype=arr[0].dtype)

        return self._ori_data

    def encrypt(self, encrypt_tool):
        return PaillierTensor(encrypt_tool.distribute_encrypt(self._obj))

    def decrypt(self, decrypt_tool):
        return PaillierTensor(self._obj.mapValues(lambda val: decrypt_tool.recursive_decrypt(val)))

    def encode(self, encoder):
        return PaillierTensor(self._obj.mapValues(lambda val: encoder.encode(val)))

    def decode(self, decoder):
        return PaillierTensor(self._obj.mapValues(lambda val: decoder.decode(val)))

    @staticmethod
    def _vector_mul(kv_iters):
        ret_mat = None
        for k, v in kv_iters:
            tmp_mat = np.outer(v[0], v[1])

            if ret_mat is not None:
                ret_mat += tmp_mat
            else:
                ret_mat = tmp_mat

        return ret_mat

    def fast_matmul_2d(self, other):
        """
        Matrix multiplication between two matrix, please ensure that self's shape is (m, n) and other's shape is (m, k)
        Their result is a matrix of (n, k)
        """
        if isinstance(other, np.ndarray):
            mat_tensor = PaillierTensor(other, partitions=self.partitions)
            return self.fast_matmul_2d(mat_tensor)

        if isinstance(other, CTableABC):
            other = PaillierTensor(other)

        func = self._vector_mul
        ret_mat = self._obj.join(other.get_obj(), lambda vec1, vec2: (vec1, vec2)).applyPartitions(func).reduce(
            lambda mat1, mat2: mat1 + mat2)

        return ret_mat

    def matmul_3d(self, other, multiply='left'):

        assert multiply in ['left', 'right']
        if isinstance(other, PaillierTensor):
            mat = other
        elif isinstance(other, CTableABC):
            mat = PaillierTensor(other)
        elif isinstance(other, np.ndarray):
            mat = PaillierTensor(other, partitions=self.partitions)
        else:
            raise ValueError('only support numpy array and Paillier Tensor')

        if multiply == 'left':
            return PaillierTensor(self._obj.join(mat._obj, lambda val1, val2: np.tensordot(val1, val2, (1, 0))),
                                  partitions=self._partitions)

        if multiply == 'right':
            return PaillierTensor(mat._obj.join(self._obj, lambda val1, val2: np.tensordot(val1, val2, (1, 0))),
                                  partitions=self._partitions)

    def element_wise_product(self, other):
        if isinstance(other, np.ndarray):
            mat = PaillierTensor(other, partitions=self.partitions)
        elif isinstance(other, CTableABC):
            mat = PaillierTensor(other)
        else:
            mat = other
        return PaillierTensor(self._obj.join(mat._obj, lambda val1, val2: val1 * val2))

    def squeeze(self, axis):
        if axis == 0:
            return PaillierTensor(list(self._obj.collect())[0][1], partitions=self.partitions)
        else:
            return PaillierTensor(self._obj.mapValues(lambda val: np.squeeze(val, axis=axis - 1)))

    def select_columns(self, select_table):
        return PaillierTensor(self._obj.join(select_table, lambda v1, v2: v1[v2]))
