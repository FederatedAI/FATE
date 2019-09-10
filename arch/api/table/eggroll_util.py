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

from arch.api import WorkMode
from eggroll.api.core import EggrollSession

_EGGROLL_CLIENT = "_eggroll_client"


def build_eggroll_session(work_mode, job_id=None, server_conf_path="eggroll/conf/server_conf.json"):
    from eggroll.api.core import EggrollSession
    import uuid
    session_id = job_id or str(uuid.uuid1())
    session = EggrollSession(session_id=session_id)
    if work_mode.is_cluster():
        from eggroll.api.cluster import eggroll
        from eggroll.api.cluster.eggroll import EGGROLL_ROLL_HOST, EGGROLL_ROLL_PORT, file_utils, getLogger
        eggroll.LOGGER = getLogger()

        server_conf = file_utils.load_json_conf(server_conf_path)
        session.add_conf('eggroll.server.conf.path', server_conf_path)
        session.add_conf(EGGROLL_ROLL_HOST, server_conf.get("servers").get("roll").get("host"))
        session.add_conf(EGGROLL_ROLL_PORT, server_conf.get("servers").get("roll").get("port"))

    return session


# noinspection PyProtectedMember
def build_eggroll_runtime(work_mode: WorkMode, eggroll_session: EggrollSession):
    if work_mode.is_standalone():
        from eggroll.api.standalone.eggroll import Standalone
        return Standalone(eggroll_session)

    elif work_mode.is_cluster():
        from eggroll.api.cluster.eggroll import _EggRoll
        return _EggRoll(eggroll_session)
    raise ValueError(f"work_mode: {work_mode} not supported!")


def broadcast_eggroll_session(sc, work_mode, eggroll_session):
    import pickle
    pickled_client = pickle.dumps((work_mode.value, eggroll_session)).hex()
    sc.setLocalProperty(_EGGROLL_CLIENT, pickled_client)


# noinspection PyProtectedMember
def maybe_create_eggroll_client():
    """
    a tricky way to set eggroll client which may be used by spark tasks.
    WARM: This may be removed or adjusted in future!
    """
    import pickle
    from pyspark.taskcontext import TaskContext
    mode, eggroll_session = pickle.loads(bytes.fromhex(TaskContext.get().getLocalProperty(_EGGROLL_CLIENT)))
    if mode == 0:
        from eggroll.api.standalone.eggroll import Standalone
        Standalone(eggroll_session)
    else:
        from eggroll.api.cluster.eggroll import _EggRoll
        _EggRoll(eggroll_session)
