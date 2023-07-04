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

logger = logging.getLogger(__name__)


def load_component(cpn_name: str):
    from fate.components.components import LazyBuildInComponentsLoader

    from .component_desc._component import Component

    # from build in
    lazy_build_in_components_loader = LazyBuildInComponentsLoader()
    if lazy_build_in_components_loader.contains(cpn_name):
        return lazy_build_in_components_loader.load_cpn(cpn_name)

    # from entrypoint
    import pkg_resources

    for cpn_ep in pkg_resources.iter_entry_points(group="fate.ext.component_desc"):
        try:
            candidate_cpn: Component = cpn_ep.load()
            candidate_cpn_name = candidate_cpn.name
        except Exception as e:
            logger.warning(
                f"register cpn from entrypoint(named={cpn_ep.name}, module={cpn_ep.module_name}) failed: {e}"
            )
            continue
        if candidate_cpn_name == cpn_name:
            return candidate_cpn
    raise RuntimeError(f"could not find registered cpn named `{cpn_name}`")


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
