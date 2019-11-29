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
import random

import numpy as np


class RandomDevice(object):
    r = random.SystemRandom()

    def __init__(self, q_field):
        self._q_field = q_field

    def rand(self, value):
        shape = value.shape
        ret = np.zeros(shape, dtype=np.int64)
        view = ret.view().reshape(-1)
        for i in range(ret.size):
            view[i] = self.r.randint(1, self._q_field)
        return ret
