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

from numpy.random import RandomState


class RandomPads(object):
    """random pads utils for secret homogeneous aggregation
    currently use numpy.random, which use bit generator MT19937
    other algorithms such as pcg and xoroshiro may be supported in the future
    """

    def __init__(self, init_seed=None):
        self._rand = RandomState(init_seed)

    def rand(self, d0, *more, **kwargs):
        return self._rand.rand(d0, *more, **kwargs)

    def randn(self, d0, *more, **kwargs):
        return self._rand.randn(d0, *more, **kwargs)

    def add_randn_pads(self, a, w):
        """a + r * w,
        where r is random array with nominal distribution N(0,1) and r.shape == a.shape
        """
        return a + self._rand.randn(*a.shape) * w

    def add_rand_pads(self, a, w):
        """a + r * w,
        where r is random array with uniform distribution U[0,1) and r.shape == a.shape
        """
        return a + self._rand.rand(*a.shape) * w
