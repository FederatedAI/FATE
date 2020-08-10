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
SIZE_LIMIT = 1 << 28  # 32MB


def split_put(k, v, use_serialize, put_call_back_func):
    if use_serialize is False:
        raise NotImplementedError("not support put large value without serialization yet!")
    v_bytes = pickle.dumps(v)
    num_bytes = len(v_bytes)
    num_splits = (num_bytes - 1) // SIZE_LIMIT + 1
    view = memoryview(v_bytes)
    put_call_back_func.put(k, num_splits, use_serialize=True)
    for i in range(num_splits):
        if use_serialize is None:
            put_call_back_func.put(k=pickle.dumps(f"{k}__frag_{i}"),
                                   v=view[SIZE_LIMIT * i: SIZE_LIMIT * (i+1)])
        else:
            put_call_back_func.put(k=pickle.dumps(f"{k}__frag_{i}"),
                                   v=view[SIZE_LIMIT * i: SIZE_LIMIT * (i + 1)],
                                   use_serialize=False)
    return True


def split_get(k, use_serialize, get_call_back_func):
    if use_serialize is False:
        raise NotImplementedError("not support get large value without serialization yet!")
    k_bytes = pickle.dumps(k)
    num_split = pickle.loads(k_bytes)
    splits = []
    for i in range(num_split):
        if use_serialize is None:
            splits.append(get_call_back_func(k=pickle.dumps(f"{k}__frag_{i}")))
        else:
            splits.append(get_call_back_func(k=pickle.dumps(f"{k}__frag_{i}"), use_serialize=False))
    v_bytes = bytes.join(*splits)
    v = pickle.loads(v_bytes)
    return v
