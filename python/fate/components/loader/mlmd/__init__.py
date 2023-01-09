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
import logging

from fate.components.spec.mlmd import (
    CustomMLMDSpec,
    FlowMLMDSpec,
    NoopMLMDSpec,
    PipelineMLMDSpec,
)

from .protocol import MLMD

logger = logging.getLogger(__name__)


def load_mlmd(mlmd, taskid) -> MLMD:
    # from buildin
    if isinstance(mlmd, PipelineMLMDSpec):
        from .pipeline import PipelineMLMD

        return PipelineMLMD(mlmd, taskid)

    if isinstance(mlmd, FlowMLMDSpec):
        from .flow import FlowMLMD

        return FlowMLMD(mlmd, taskid)

    if isinstance(mlmd, NoopMLMDSpec):
        from .noop import NoopMLMD

        return NoopMLMD(mlmd, taskid)
    # from entrypoint
    if isinstance(mlmd, CustomMLMDSpec):
        import pkg_resources

        for mlmd_ep in pkg_resources.iter_entry_points(group="fate.ext.mlmd"):
            try:
                mlmd_register = mlmd_ep.load()
                mlmd_registered_name = mlmd_register.registered_name()
            except Exception as e:
                logger.warning(
                    f"register cpn from entrypoint(named={mlmd_ep.name}, module={mlmd_ep.module_name}) failed: {e}"
                )
                continue
            if mlmd_registered_name == mlmd.name:
                return mlmd_register
        raise RuntimeError(f"could not find registerd mlmd named `{mlmd.name}`")

    raise ValueError(f"unknown mlmd spec: `{mlmd}`")
