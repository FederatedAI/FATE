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


def partitioner(hash_func, total_partitions):
    def partition(key):
        return hash_func(key) % total_partitions

    return partition


def get_default_partitioner():
    from ._mmh3_partitioner import mmh3_partitioner

    return mmh3_partitioner


def get_partitioner_by_type(partitioner_type: int):
    if partitioner_type == 0:
        return get_default_partitioner()
    elif partitioner_type == 1:
        from ._integer_partitioner import integer_partitioner

        return integer_partitioner
    elif partitioner_type == 2:
        from ._mmh3_partitioner import mmh3_partitioner

        return mmh3_partitioner
    elif partitioner_type == 3:
        from ._java_string_like_partitioner import _java_string_like_partitioner

        return _java_string_like_partitioner
    else:
        raise ValueError(f"partitioner type `{partitioner_type}` not supported")
