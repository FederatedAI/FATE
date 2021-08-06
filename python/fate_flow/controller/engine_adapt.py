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
from fate_flow.controller.engine_operation.eggroll import EggrollEngine
from fate_flow.controller.engine_operation.linkis_spark import LinkisSparkEngine
from fate_flow.controller.engine_operation.spark import SparkEngine


def build_engine(computing_engine):
    if computing_engine in {ComputingEngine.EGGROLL, ComputingEngine.STANDALONE}:
        engine_session = EggrollEngine()
    elif computing_engine == ComputingEngine.SPARK:
        engine_session = SparkEngine()
    elif computing_engine == ComputingEngine.LINKIS_SPARK:
        engine_session = LinkisSparkEngine()
    else:
        raise ValueError(f"${computing_engine} is not supported")
    return engine_session
