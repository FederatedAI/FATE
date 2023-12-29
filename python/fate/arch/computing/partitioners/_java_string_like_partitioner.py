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

import hashlib


def _java_string_like_partitioner(key, total_partitions):
    _key = hashlib.sha1(key).digest()
    _key = int.from_bytes(_key, byteorder="little", signed=False)
    b, j = -1, 0
    while j < total_partitions:
        b = int(j)
        _key = ((_key * 2862933555777941757) + 1) & 0xFFFFFFFFFFFFFFFF
        j = float(b + 1) * (float(1 << 31) / float((_key >> 33) + 1))
    return int(b)
