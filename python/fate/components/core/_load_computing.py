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
def load_computing(computing, logger_config=None):
    from fate.components.core.spec.computing import (
        EggrollComputingSpec,
        SparkComputingSpec,
        StandaloneComputingSpec,
    )

    if isinstance(computing, StandaloneComputingSpec):
        from fate.arch.computing.standalone import CSession

        return CSession(
            computing.metadata.computing_id, logger_config=logger_config, options=computing.metadata.options
        )
    if isinstance(computing, EggrollComputingSpec):
        from fate.arch.computing.eggroll import CSession

        return CSession(computing.metadata.computing_id, options=computing.metadata.options)
    if isinstance(computing, SparkComputingSpec):
        from fate.arch.computing.spark import CSession

        return CSession(computing.metadata.computing_id)

    # TODO: load from plugin
    raise ValueError(f"conf.computing={computing} not support")
