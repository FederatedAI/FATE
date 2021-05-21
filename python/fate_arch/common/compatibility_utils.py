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
import typing

from fate_arch.common import WorkMode, Backend, FederatedMode
from fate_arch.computing import ComputingEngine
from fate_arch.federation import FederationEngine


def backend_compatibility(work_mode: typing.Union[WorkMode, int] = WorkMode.STANDALONE,
                          backend: typing.Union[Backend, int] = Backend.EGGROLL, **kwargs):
    # Compatible with previous 1.5 versions
    if kwargs.get("computing_engine") is None or kwargs.get("federation_engine") is None or kwargs.get(
            "federation_mode") is None:
        if work_mode is None or backend is None:
            raise RuntimeError("unable to find compatible engines")
        if isinstance(work_mode, int):
            work_mode = WorkMode(work_mode)
        if isinstance(backend, int):
            backend = Backend(backend)
        if backend == Backend.EGGROLL:
            if work_mode == WorkMode.CLUSTER:
                return ComputingEngine.EGGROLL, FederationEngine.EGGROLL, FederatedMode.MULTIPLE
            else:
                return ComputingEngine.STANDALONE, FederationEngine.STANDALONE, FederatedMode.SINGLE
        if backend == Backend.SPARK_RABBITMQ:
            return ComputingEngine.SPARK, FederationEngine.RABBITMQ, FederatedMode.MULTIPLE
        if backend == Backend.SPARK_PULSAR:
            return ComputingEngine.SPARK, FederationEngine.PULSAR, FederatedMode.MULTIPLE
    else:
        return kwargs["computing_engine"], kwargs["federation_engine"], kwargs["federated_mode"]
