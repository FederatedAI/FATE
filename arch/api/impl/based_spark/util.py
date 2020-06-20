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

import random, string


RDD_ATTR_NAME = "_rdd"


# noinspection PyUnresolvedReferences
def get_storage_level():
    from pyspark import StorageLevel
    return StorageLevel.MEMORY_AND_DISK


def materialize(rdd):
    rdd.persist(get_storage_level())
    rdd.mapPartitionsWithIndex(lambda ind, it: (1,)).collect()
    return rdd


def RandomString(stringLength=6):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def RandomNumberString(stringLength=6):
    letters = string.octdigits
    return ''.join(random.choice(letters) for i in range(stringLength))

