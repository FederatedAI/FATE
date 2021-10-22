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

from fate_arch.common import WorkMode,  FederatedMode, conf_utils
from fate_arch.computing import ComputingEngine
from fate_arch.federation import FederationEngine
from fate_arch.storage import StorageEngine
from fate_arch.relation_ship import Relationship
from fate_arch.common import EngineType


def get_engine_class_members(engine_class) -> list:
    members = []
    for k, v in engine_class.__dict__.items():
        if k in ["__module__", "__dict__", "__weakref__", "__doc__"]:
            continue
        members.append(v)
    return members


def get_engines(work_mode: typing.Union[WorkMode, int] = None, options=None):
    keys = [EngineType.COMPUTING, EngineType.FEDERATION, EngineType.STORAGE, "federated_mode"]
    engines = {}
    for k in keys:
        if options and options.get(k, None) is not None:
            engines[k] = options[k].upper()

    if isinstance(work_mode, int):
        work_mode = WorkMode(work_mode)

    if work_mode == WorkMode.STANDALONE:
        engines[EngineType.COMPUTING] = ComputingEngine.STANDALONE
        engines[EngineType.FEDERATION] = FederationEngine.STANDALONE
        engines[EngineType.STORAGE] = StorageEngine.STANDALONE

    if engines.get(EngineType.COMPUTING) is None:
        if conf_utils.get_base_config("default_engines", {}).get(EngineType.COMPUTING) is not None:
            default_engines = conf_utils.get_base_config("default_engines")

            engines[EngineType.COMPUTING] = default_engines[EngineType.COMPUTING].upper() \
                if default_engines.get(EngineType.COMPUTING) is not None else None

            if engines.get(EngineType.FEDERATION) is None:
                engines[EngineType.FEDERATION] = default_engines[EngineType.FEDERATION].upper() \
                    if default_engines.get(EngineType.FEDERATION) is not None else None

            if engines.get(EngineType.STORAGE) is None:
                engines[EngineType.STORAGE] = default_engines[EngineType.STORAGE].upper() \
                    if default_engines.get(EngineType.STORAGE) is not None else None
        else:
            raise RuntimeError(f"must set default_engines on conf/service_conf.yaml")

    if engines.get(EngineType.COMPUTING) is None:
        raise RuntimeError(f"{EngineType.COMPUTING} is None,"
                           f"Please check work_mode parameter or default_engines on conf/service_conf.yaml")

    if engines[EngineType.COMPUTING] not in get_engine_class_members(ComputingEngine):
        raise RuntimeError(f"{engines[EngineType.COMPUTING]} is illegal")

    # set default storage engine and federation engine by computing engine
    for t in (EngineType.STORAGE, EngineType.FEDERATION):
        if engines.get(t) is None:
            # use default relation engine
            engines[t] = Relationship.Computing[engines[EngineType.COMPUTING]][t]["default"]

    # set default federated mode by federation engine
    if engines.get("federated_mode") is None:
        if engines[EngineType.FEDERATION] == FederationEngine.STANDALONE:
            engines["federated_mode"] = FederatedMode.SINGLE
        else:
            engines["federated_mode"] = FederatedMode.MULTIPLE

    if engines[EngineType.STORAGE] not in get_engine_class_members(StorageEngine):
        raise RuntimeError(f"{engines[EngineType.STORAGE]} is illegal")

    if engines[EngineType.FEDERATION] not in get_engine_class_members(FederationEngine):
        raise RuntimeError(f"{engines[EngineType.FEDERATION]} is illegal")

    for t in [EngineType.FEDERATION]:
        if engines[t] not in Relationship.Computing[engines[EngineType.COMPUTING]][t]["support"]:
            raise RuntimeError(f"{engines[t]} is not supported in {engines[EngineType.COMPUTING]}")

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
