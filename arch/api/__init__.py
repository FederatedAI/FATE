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


from enum import IntEnum

from arch.api.base.build import Builder
from arch.api.base.federation import Federation
from arch.api.base.utils.wrap import FederationWrapped

_EGGROLL_VERSION = 2


class WorkMode(IntEnum):
    STANDALONE = 0
    CLUSTER = 1

    def is_standalone(self):
        return self.value == self.STANDALONE

    def is_cluster(self):
        return self.value == self.CLUSTER


class Backend(IntEnum):
    EGGROLL = 0
    SPARK = 1

    def is_spark(self):
        return self.value == self.SPARK

    def is_eggroll(self):
        return self.value == self.EGGROLL


class RuntimeInstance(object):
    SESSION = None
    MODE: WorkMode = None
    FEDERATION: Federation = None
    TABLE_WRAPPER: FederationWrapped = None
    BACKEND: Backend = None
    BUILDER: Builder = None
