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

from arch.api import session
from arch.api.utils import log_utils

LOGGER = log_utils.getLogger()


class PaillierTensor(object):
    def __init__(self, ori_data=None, tb_obj=None, partitions=1):
        if ori_data is not None:
            self._ori_data = ori_data
            self._partitions = partitions
            self._obj = session.parallelize(ori_data,
                                            include_key=False,
                                            partition=partitions)
        else:
            self._ori_data = None
            self._partitions = tb_obj._partitions
            self._obj = tb_obj

        LOGGER.debug("tensor's partition is {}".format(self._partitions))

    def __add__(self, other):
        if isinstance(other, PaillierTensor):
            return PaillierTensor(tb_obj=self._obj.join(other._obj, lambda v1, v2: v1 + v2))
        else:
            return PaillierTensor(tb_obj=self._obj.mapValues(lambda v: v + other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, PaillierTensor):
            return PaillierTensor(tb_obj=self._obj.join(other._obj, lambda v1, v2: v1 - v2))
        else:
            return PaillierTensor(tb_obj=self._obj.mapValues(lambda v: v - other))

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return PaillierTensor(tb_obj=self._obj.mapValues(lambda val: val * other))
        elif isinstance(other, np.ndarray):
            return PaillierTensor(tb_obj=self._obj.mapValues(lambda val: np.matmul(val, other)))

        try:
            _other = other.numpy()
        except AttributeError:
            raise ValueError("multiply does not support")

        ret = self.numpy() * _other

        return PaillierTensor(ori_data=ret, partitions=max(self.partitions, other.partitions))

    def multiply(self, other):
        if not isinstance(other, PaillierTensor):
            raise ValueError("multiply operator of HeteroNNTensor should between to HeteroNNTensor")

        return PaillierTensor(tb_obj=self._obj.join(other._obj, lambda val1, val2: np.multiply(val1, val2)),
                              partitions=self._partitions)

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

            return PaillierTensor(tb_obj=ret_obj)

    def numpy(self):
        if self._ori_data is not None:
            return self._ori_data

        arr = [None for i in range(self._obj.count())]

        for k, v in self._obj.collect():
            arr[k] = v

        self._ori_data = np.array(arr, dtype=arr[0].dtype)

        return self._ori_data

    def encrypt(self, encrypt_tool):
        return PaillierTensor(tb_obj=encrypt_tool.encrypt(self._obj))

    def decrypt(self, decrypt_tool):
        return PaillierTensor(tb_obj=self._obj.mapValues(lambda val: decrypt_tool.recursive_decrypt(val)))

    @staticmethod
    def _vector_mul(kv_iters):
        ret_mat = None
        for k, v in kv_iters:
            tmp_mat = np.tensordot(v[0], v[1], [[], []])

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
            mat_tensor = PaillierTensor(ori_data=other, partitions=self.partitions)
            return self.fast_matmul_2d(mat_tensor)

        func = self._vector_mul
        ret_mat = self._obj.join(other.get_obj(), lambda vec1, vec2: (vec1, vec2)).mapPartitions(func).reduce(
            lambda mat1, mat2: mat1 + mat2)

        return ret_mat
