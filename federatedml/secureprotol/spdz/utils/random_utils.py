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

from arch.api.base.table import Table


def rand_tensor(q_field, tensor):
    if isinstance(tensor, Table):
        return tensor.mapValues(
            lambda x: np.random.randint(1, q_field, len(x)).astype(object))
    if isinstance(tensor, np.ndarray):
        arr = np.random.randint(1, q_field, tensor.shape).astype(object)
        return arr
    raise NotImplementedError(f"type={type(tensor)}")


def urand_tensor(q_field, tensor):
    if isinstance(tensor, Table):
        return tensor.mapValues(
            lambda x: np.array([random.SystemRandom().randint(1, q_field) for _ in x], dtype=object))
    if isinstance(tensor, np.ndarray):
        arr = np.zeros(shape=tensor.shape, dtype=object)
        view = arr.view().reshape(-1)
        for i in range(arr.size):
            view[i] = random.SystemRandom().randint(1, q_field)
        return arr
    raise NotImplementedError(f"type={type(tensor)}")
