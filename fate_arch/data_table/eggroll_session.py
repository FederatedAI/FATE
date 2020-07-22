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
import uuid

from eggroll.core.constants import SerdesTypes
from eggroll.core.session import session_init
from eggroll.roll_pair.roll_pair import RollPairContext
from fate_arch.common import WorkMode
from fate_arch.common.profile import log_elapsed


def get_session(session_id='', work_mode: int = 0, options: dict = None):
    if not session_id:
        session_id = str(uuid.uuid1())
    return Session(session_id=session_id, work_mode=work_mode, options=options)


class Session(object):
    def __init__(self, session_id, work_mode, options: dict = None):
        if options is None:
            options = {}
        if work_mode == WorkMode.STANDALONE:
            options['eggroll.session.deploy.mode'] = "standalone"
        elif work_mode == WorkMode.CLUSTER:
            options['eggroll.session.deploy.mode'] = "cluster"
        self._rp_session = session_init(session_id=session_id, options=options)
        self._rpc = RollPairContext(session=self._rp_session)
        self._session_id = self._rp_session.get_session_id()

    def table(self,
              name,
              namespace,
              partition,
              create_if_missing,
              **kwargs):
        options = kwargs.get("option", {})
        if "use_serialize" in kwargs and not kwargs["use_serialize"]:
            options["serdes"] = SerdesTypes.EMPTY
        options.update(dict(create_if_missing=create_if_missing, total_partitions=partition))
        _table = self._rpc.load(namespace=namespace, name=name, options=options)
        return _table

    def _get_session_id(self):
        return self._session_id

    @log_elapsed
    def cleanup(self, name, namespace):
        self._rpc.cleanup(name=name, namespace=namespace)

    @log_elapsed
    def stop(self):
        return self._rp_session.stop()

    @log_elapsed
    def kill(self):
        return self._rp_session.kill()
