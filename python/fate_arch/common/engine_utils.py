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

from fate_arch.common import WorkMode, Backend, FederatedMode, conf_utils
from fate_arch.computing import ComputingEngine
from fate_arch.federation import FederationEngine
from fate_arch.storage import StorageEngine
from fate_arch.relation_ship import Relationship
from fate_arch.common import EngineType
from fate_arch.common.conf_utils import get_base_config


def engines_compatibility(work_mode: typing.Union[WorkMode, int] = None,
                          backend: typing.Union[Backend, int] = None, **kwargs):
    keys = ["computing", "federation", "storage", "federated_mode"]
    engines = {}
    for k in keys:
        if kwargs.get(k) is not None:
            engines[k] = kwargs[k]
    if kwargs.get("computing") is None and work_mode is None:
        raise RuntimeError("must provide computing engine parameters or work_mode parameters")
    if kwargs.get("computing") is None and kwargs.get("federation") is None:
        if isinstance(work_mode, int):
            work_mode = WorkMode(work_mode)
        if backend is not None:
            if isinstance(backend, int):
                backend = Backend(backend)
            if backend == Backend.EGGROLL:
                if work_mode == WorkMode.CLUSTER:
                    values = (ComputingEngine.EGGROLL, FederationEngine.EGGROLL)
                else:
                    values = (ComputingEngine.STANDALONE, FederationEngine.STANDALONE)
            elif backend == Backend.SPARK_RABBITMQ:
                values = (ComputingEngine.SPARK, FederationEngine.RABBITMQ)
            elif backend == Backend.SPARK_PULSAR:
                values = (ComputingEngine.SPARK, FederationEngine.PULSAR)
            elif backend == Backend.LINKIS_SPARK_RABBITMQ:
                values = (ComputingEngine.LINKIS_SPARK, FederationEngine.RABBITMQ)
            else:
                raise RuntimeError(f"unable to find default engines by work_mode: {work_mode} backend: {backend}")
        elif get_base_config("default_engines", {}).get("computing"):
            default_engines = get_base_config("default_engines")
            values = (default_engines["computing"].upper(), default_engines["federation"].upper() if "federation" in default_engines else None)
            engines["storage"] = default_engines.get("storage", None)
        else:
            raise RuntimeError(f"must provide backend or set default engines on conf/service_conf.yaml")
        engines.update(dict(zip(keys[:2], values)))

    # set default storage engine and federation engine by computing engine
    for t in {EngineType.STORAGE, EngineType.FEDERATION}:
        if engines.get(t) is None:
            # use default relation engine
            engines[t] = Relationship.Computing[engines[EngineType.COMPUTING]][t]["default"]

    # set default federated mode by federation engine
    if engines.get("federated_mode") is None:
        if engines[EngineType.FEDERATION] == FederationEngine.STANDALONE:
            engines["federated_mode"] = FederatedMode.SINGLE
        else:
            engines["federated_mode"] = FederatedMode.MULTIPLE

    return engines


def get_engines_config_from_conf(group_map=False):
    engines_config = {}
    engine_group_map = {}
    for engine_type in {EngineType.COMPUTING, EngineType.FEDERATION, EngineType.STORAGE}:
        engines_config[engine_type] = {}
        engine_group_map[engine_type] = {}
    for group_name, engine_map in Relationship.EngineConfMap.items():
        for engine_type, name_maps in engine_map.items():
            for name_map in name_maps:
                single_engine_config = conf_utils.get_base_config(group_name, {}).get(name_map[1], {})
                if single_engine_config:
                    engine_name = name_map[0]
                    engines_config[engine_type][engine_name] = single_engine_config
                    engine_group_map[engine_type][engine_name] = group_name
    if not group_map:
        return engines_config
    else:
        return engines_config, engine_group_map
