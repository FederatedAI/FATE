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
from fate_arch.computing import ComputingEngine
from fate_arch.federation import FederationEngine
from fate_arch.storage import StorageEngine
from fate_arch.common.address import StandaloneAddress, EggRollAddress, HDFSAddress, \
    MysqlAddress, \
    PathAddress, LocalFSAddress, HiveAddress, LinkisHiveAddress
from fate_arch.common import EngineType


class Relationship(object):
    Computing = {
        ComputingEngine.STANDALONE: {
            EngineType.STORAGE: {
                "default": StorageEngine.STANDALONE,
                "support": [StorageEngine.STANDALONE]
            },
            EngineType.FEDERATION: {
                "default": FederationEngine.STANDALONE,
                "support": [FederationEngine.STANDALONE]
            },
        },
        ComputingEngine.EGGROLL: {
            EngineType.STORAGE: {
                "default": StorageEngine.EGGROLL,
                "support": [StorageEngine.EGGROLL]
            },
            EngineType.FEDERATION: {
                "default": FederationEngine.EGGROLL,
                "support": [FederationEngine.EGGROLL]
            },
        },
        ComputingEngine.SPARK: {
            EngineType.STORAGE: {
                "default": StorageEngine.HDFS,
                "support": [StorageEngine.HDFS, StorageEngine.HIVE, StorageEngine.LOCALFS]
            },
            EngineType.FEDERATION: {
                "default": FederationEngine.RABBITMQ,
                "support": [FederationEngine.PULSAR, FederationEngine.RABBITMQ]
            },
        },
        ComputingEngine.LINKIS_SPARK: {
            EngineType.STORAGE: {
                "default": StorageEngine.LINKIS_HIVE,
                "support": [StorageEngine.LINKIS_HIVE]
            },
            EngineType.FEDERATION: {
                "default": FederationEngine.RABBITMQ,
                "support": [FederationEngine.PULSAR, FederationEngine.RABBITMQ]
            },
        }
    }

    EngineToAddress = {
        StorageEngine.STANDALONE: StandaloneAddress,
        StorageEngine.EGGROLL: EggRollAddress,
        StorageEngine.HDFS: HDFSAddress,
        StorageEngine.MYSQL: MysqlAddress,
        StorageEngine.HIVE: HiveAddress,
        StorageEngine.LINKIS_HIVE: LinkisHiveAddress,
        StorageEngine.LOCALFS: LocalFSAddress,
        StorageEngine.PATH: PathAddress
    }

    EngineConfMap = {
        "fate_on_standalone": {
            EngineType.COMPUTING: [(ComputingEngine.STANDALONE, "standalone")],
            EngineType.STORAGE: [(StorageEngine.STANDALONE, "standalone")],
            EngineType.FEDERATION: [(FederationEngine.STANDALONE, "standalone")]
        },
        "fate_on_eggroll": {
            EngineType.COMPUTING: [(ComputingEngine.EGGROLL, "clustermanager")],
            EngineType.STORAGE: [(StorageEngine.EGGROLL, "clustermanager")],
            EngineType.FEDERATION: [(FederationEngine.EGGROLL, "rollsite")],
        },
        "fate_on_spark": {
            EngineType.COMPUTING: [(ComputingEngine.SPARK, "spark"), (ComputingEngine.LINKIS_SPARK, "linkis_spark")],
            EngineType.STORAGE: [(StorageEngine.HDFS, "hdfs"), (StorageEngine.HIVE, "hive"),
                                 (StorageEngine.LINKIS_HIVE, "linkis_hive"), (StorageEngine.LOCALFS, "localfs")],
            EngineType.FEDERATION: [(FederationEngine.RABBITMQ, "rabbitmq"), (FederationEngine.PULSAR, "pulsar")]
        },
    }
