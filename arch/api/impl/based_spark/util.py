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

_EGGROLL_CLIENT = "_eggroll_client"
RDD_ATTR_NAME = "_rdd"


# noinspection PyUnresolvedReferences
def get_storage_level():
    from pyspark import StorageLevel
    return StorageLevel.MEMORY_AND_DISK


def materialize(rdd):
    rdd.persist(get_storage_level())
    rdd.mapPartitionsWithIndex(lambda ind, it: (1,)).collect()
    return rdd


# noinspection PyUnresolvedReferences
def broadcast_eggroll_session(work_mode, eggroll_session):
    import pickle
    pickled_client = pickle.dumps((work_mode.value, eggroll_session)).hex()
    from pyspark import SparkContext
    SparkContext.getOrCreate().setLocalProperty(_EGGROLL_CLIENT, pickled_client)


# noinspection PyProtectedMember,PyUnresolvedReferences
def maybe_create_eggroll_client():
    """
    a tricky way to set eggroll client which may be used by spark tasks.
    WARM: This may be removed or adjusted in future!
    """
    import pickle
    from pyspark.taskcontext import TaskContext
    mode, eggroll_session = pickle.loads(bytes.fromhex(TaskContext.get().getLocalProperty(_EGGROLL_CLIENT)))

    from arch.api import _EGGROLL_VERSION
    if _EGGROLL_VERSION < 2:
        if mode == 1:
            from eggroll.api.cluster.eggroll import _EggRoll
            if _EggRoll.instance is None:
                from eggroll.api import ComputingEngine
                from eggroll.api.cluster.eggroll import _EggRoll
                eggroll_runtime = _EggRoll(eggroll_session=eggroll_session)
                eggroll_session.set_runtime(ComputingEngine.EGGROLL_DTABLE, eggroll_runtime)
        else:
            from eggroll.api.standalone.eggroll import Standalone
            Standalone(eggroll_session)
