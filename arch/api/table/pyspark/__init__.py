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

from pyspark import RDD
from pyspark.storagelevel import StorageLevel

# noinspection PyUnresolvedReferences
STORAGE_LEVEL = StorageLevel.MEMORY_AND_DISK

_RDD_ATTR_NAME = "_rdd"


def materialize(rdd: RDD):
    rdd.persist(STORAGE_LEVEL)
    rdd.mapPartitionsWithIndex(lambda ind, it: (1,)).collect()
    return rdd


class JobDesc(object):
    def __init__(self, desc, **kwargs):
        if "msg" in kwargs:
            self._desc = f"{desc}: {kwargs['msg']}"
        else:
            self._desc = desc

    def __enter__(self):
        import pyspark
        pyspark.SparkContext.getOrCreate().setLocalProperty("spark.job.description", self._desc)

    def __exit__(self, exc_type, exc_val, exc_tb):
        import pyspark
        pyspark.SparkContext.getOrCreate().setLocalProperty("spark.job.description", "")


__all__ = ["STORAGE_LEVEL", "materialize", "_RDD_ATTR_NAME", "JobDesc"]
