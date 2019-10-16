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

from arch.api import RuntimeInstance
from arch.api.transfer import FederationBuilder
from arch.api.utils.profile_util import log_elapsed


def init(job_id, runtime_conf, server_conf_path="arch/conf/server_conf.json"):
    """
    This method is required before get/remote called.
    :param job_id: current job_id, None is ok, uuid will be used.
    :param runtime_conf:
    :param server_conf_path:
    runtime_conf should be a dict with
     1. key "local" maps to the current process' role and party_id.
     2. key "role" maps to a dict mapping from each role to all involving party_ids.
     {
        "local": {
            "role": "host",
            "party_id": 1000
        }
        "role": {
            "host": [999, 1000, 1001],
            "guest": [10002]
        }
     }

    """
    if RuntimeInstance.MODE is None:
        raise EnvironmentError(
            "table manager should be initialized before federation")

    if RuntimeInstance.Backend.is_eggroll():
        RuntimeInstance.FEDERATION = \
            FederationBuilder.build_eggroll_backend(job_id=job_id,
                                                    runtime_conf=runtime_conf,
                                                    work_mode=RuntimeInstance.MODE,
                                                    server_conf_path=server_conf_path)

    elif RuntimeInstance.Backend.is_spark():
        RuntimeInstance.FEDERATION = \
            FederationBuilder.build_spark_backend(job_id=job_id,
                                                  runtime_conf=runtime_conf,
                                                  work_mode=RuntimeInstance.MODE,
                                                  server_conf_path=server_conf_path)


@log_elapsed
def get(name, tag: str, idx=-1):
    """
    This method will block until the remote object is fetched.
    :param name: {alogrithm}.{variableName} defined in transfer_conf.json.
    :param tag: object version, should be a string.
    :param idx: idx of the party_ids in runtime role list, if out-of-range, list of all objects will be returned.
    :return: The object itself if idx is in range, else return list of all objects from possible source.
    """

    return RuntimeInstance.FEDERATION.get(name=name, tag=tag, idx=idx)


@log_elapsed
def remote(obj, name: str, tag: str, role=None, idx=-1):
    """
    This method will send an object to other parties
    :param obj: The object itself which can be pickled.
    :param name: {alogrithm}.{variableName} defined in transfer_conf.json.
    :param tag: tag: object version, should be a string.
    :param role: The role you want to send to.
    :param idx: The idx of the party_ids of the role, if out-of-range, will send to all parties of the role.
    :return: None
    """
    return RuntimeInstance.FEDERATION.remote(obj=obj,
                                             name=name,
                                             tag=tag,
                                             role=role,
                                             idx=idx)


def get_runtime_conf():
    return RuntimeInstance.FEDERATION.runtime_conf
