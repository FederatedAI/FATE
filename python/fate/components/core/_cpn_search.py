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
import typing
from typing import Optional

if typing.TYPE_CHECKING:
    from fate.components.core import Component, Stage

logger = logging.getLogger(__name__)


def load_component(cpn_name: str, stage: Optional["Stage"] = None):
    from fate.components.components import LazyBuildInComponentsLoader

    # from build in
    cpn = None
    lazy_build_in_components_loader = LazyBuildInComponentsLoader()
    if lazy_build_in_components_loader.contains(cpn_name):
        cpn = lazy_build_in_components_loader.load_cpn(cpn_name)
    else:
        # from entrypoint
        import pkg_resources

        for cpn_ep in pkg_resources.iter_entry_points(group="fate.ext.component_desc"):
            try:
                candidate_cpn: "Component" = cpn_ep.load()
                candidate_cpn_name = candidate_cpn.name
            except Exception as e:
                logger.warning(
                    f"register cpn from entrypoint(named={cpn_ep.name}, module={cpn_ep.module_name}) failed: {e}"
                )
                continue
            if candidate_cpn_name == cpn_name:
                cpn = candidate_cpn
                break
    if cpn is None:
        raise RuntimeError(f"could not find registered cpn named `{cpn_name}`")
    if stage is not None:
        cpn = load_stage_component(cpn, stage)
    return cpn


def load_stage_component(cpn, stage: "Stage"):
    if not stage.is_default:
        for stage_component in cpn.stage_components:
            if stage_component.name == stage.name:
                cpn = stage_component
                break
        else:
            supported_stage_names = [stage_component.name for stage_component in cpn.stage_components]
            raise ValueError(
                f"stage `{stage.name}` not supported for component `{cpn.name}`, use one listed in: {supported_stage_names}"
            )
    return cpn


def list_components():
    import pkg_resources
    from fate.components.components import LazyBuildInComponentsLoader

    build_in_components = LazyBuildInComponentsLoader().list()
    third_parties_components = []

    for cpn_ep in pkg_resources.iter_entry_points(group="fate.ext.component_desc"):
        try:
            candidate_cpn = cpn_ep.load()
            candidate_cpn_name = candidate_cpn.name
            third_parties_components.append([candidate_cpn_name])
        except Exception as e:
            logger.warning(
                f"register cpn from entrypoint(named={cpn_ep.name}, module={cpn_ep.module_name}) failed: {e}"
            )
            continue
    return dict(buildin=build_in_components, thirdparty=third_parties_components)
