#
#  Copyright 2019 The Eggroll Authors. All Rights Reserved.
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

from enum import IntEnum, Enum


class WorkMode(IntEnum):
    STANDALONE = 0
    CLUSTER = 1
    SIMPLE = 2


class RuntimeInstance(object):
    EGGROLL = None
    MODE = None
    CLUSTERCOMM = None


class StoreType(Enum):
    IN_MEMORY = "IN_MEMORY"
    LMDB = "LMDB"
    LEVEL_DB = "LEVEL_DB"


class NamingPolicy(Enum):
    DEFAULT = 'DEFAULT'
    ITER_AWARE = 'ITER_AWARE'


class ComputingEngine(Enum):
    EGGROLL = 'EGGROLL'
    EGGROLL_DTABLE = 'EGGROLL_DTABLE'
