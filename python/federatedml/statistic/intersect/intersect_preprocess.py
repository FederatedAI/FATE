#
#  Copyright 2021 The FATE Authors. All Rights Reserved.
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

import gmpy2
import math
import uuid
import numpy as np

from federatedml.secureprotol.hash.hash_factory import Hash
from federatedml.util import consts, LOGGER

SALT_LENGTH = 8


class BitArray(object):
    def __init__(self, bit_count, hash_func_count, hash_method, random_state, salt=None):
        self.bit_count = bit_count
        self._array = np.zeros((bit_count + 63) // 64, dtype='uint64')
        self.bit_count = self._array.size * 64
        self.random_state = random_state
        # self.hash_encoder = Hash(hash_method, False)
        self.hash_method = hash_method
        self.hash_func_count = hash_func_count
        self.id = str(uuid.uuid4())
        self.salt = salt
        if salt is None:
            self.salt = self._generate_salt()

    def _generate_salt(self):
        random_state = np.random.RandomState(self.random_state)

        def f(n):
            return str(n)[2:]
        return list(map(f, np.round(random_state.random(self.hash_func_count), SALT_LENGTH)))

    @property
    def sparsity(self):
        set_bit_count = sum(map(gmpy2.popcount, map(int, self._array)))
        return 1 - set_bit_count / self.bit_count

    def set_array(self, new_array):
        self._array = new_array

    def get_array(self):
        return self._array

    def merge_filter(self, other):
        if self.bit_count != other.bit_count:
            raise ValueError(f"cannot merge filters with different bit count")
        self._array |= other._array

    def get_ind_set(self, x):
        hash_encoder = Hash(self.hash_method, False)
        return set(int(hash_encoder.compute(x,
                                            suffix_salt=self.salt[i]),
                       16) % self.bit_count for i in range(self.hash_func_count))

    def insert(self, x):
        """
        insert given instance to bit array with hash functions
        Parameters
        ----------
        x

        Returns
        -------

        """
        ind_set = self.get_ind_set(x)
        for ind in ind_set:
            self.set_bit(ind)
        return self._array

    def insert_ind_set(self, ind_set):
        """
        insert given ind collection to bit array with hash functions
        Parameters
        ----------
        ind_set

        Returns
        -------

        """
        for ind in ind_set:
            self.set_bit(ind)

    def check(self, x):
        """
        check whether given instance x exists in bit array
        Parameters
        ----------
        x

        Returns
        -------

        """
        hash_encoder = Hash(self.hash_method, False)
        for i in range(self.hash_func_count):
            ind = int(hash_encoder.compute(x, suffix_salt=self.salt[i]), 16) % self.bit_count
            if not self.query_bit(ind):
                return False
        return True

    def check_ind_set(self, ind_set):
        """
        check whether all bits in given ind set are filled
        Parameters
        ----------
        ind_set

        Returns
        -------

        """
        for ind in ind_set:
            if not self.query_bit(ind):
                return False
        return True

    def set_bit(self, ind):
        """
        set bit at given bit index
        Parameters
        ----------
        ind

        Returns
        -------

        """

        pos = ind >> 6
        bit_pos = ind & 63
        self._array[pos] |= np.uint64(1 << bit_pos)

    def query_bit(self, ind):
        """
        query bit != 0
        Parameters
        ----------
        ind

        Returns
        -------

        """
        pos = ind >> 6
        bit_pos = ind & 63
        return (self._array[pos] & np.uint64(1 << bit_pos)) != 0

    @staticmethod
    def get_filter_param(n, p):
        """

        Parameters
        ----------
        n: items to store in filter
        p: target false positive rate

        Returns
        -------

        """
        # bit count
        m = math.ceil(-n * math.log(p) / (math.pow(math.log(2), 2)))
        # hash func count
        k = round(m / n * math.log(2))
        if k < consts.MIN_HASH_FUNC_COUNT:
            LOGGER.info(f"computed k value {k} is smaller than min hash func count limit, "
                        f"set to {consts.MIN_HASH_FUNC_COUNT}")
            k = consts.MIN_HASH_FUNC_COUNT
            # update bit count so that target fpr = p
            m = round(-n * k / math.log(1 - math.pow(p, 1 / k)))

        if k > consts.MAX_HASH_FUNC_COUNT:
            LOGGER.info(f"computed k value {k} is greater than max hash func count limit, "
                        f"set to {consts.MAX_HASH_FUNC_COUNT}")
            k = consts.MAX_HASH_FUNC_COUNT
            # update bit count so that target fpr = p
            m = round(-n * k / math.log(1 - math.pow(p, 1 / k)))
        return m, k
