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
import importlib

from fate_flow.entity.component_provider import ComponentProvider
from fate_flow.db.runtime_config import RuntimeConfig


def component_provider(provider_info):
    name, version = provider_info["name"], provider_info["version"]
    path = RuntimeConfig.COMPONENT_REGISTRY["provider"].get(name, {}).get(version, {}).get("path", [])
    return ComponentProvider(name=name, version=version, path=path)


def get_job_provider_group(dsl_parser, role, party_id, component_name=None):
    providers = dsl_parser.get_job_providers(provider_detail=RuntimeConfig.COMPONENT_REGISTRY,
                                             local_role=role,
                                             local_party_id=party_id)
    # providers format: {'upload_0': {'module': 'Upload', 'provider': {'name': 'fate_flow_tools', 'version': '1.7.0'}}}

    group = {}
    if component_name is not None:
        providers = {component_name: providers.get(component_name)}
    for component_name, provider_info in providers.items():
        provider = component_provider(provider_info["provider"])
        group_key = ":".join([provider.name, provider.version])
        if group_key not in group:
            group[group_key] = {
                "provider": provider.to_json(),
                "components": [component_name]
            }
        else:
            group[group_key]["components"].append(component_name)
    return group


def get_component_provider(dsl_parser, component_name, role, party_id):
    providers = dsl_parser.get_job_providers(provider_detail=RuntimeConfig.COMPONENT_REGISTRY,
                                             local_role=role,
                                             local_party_id=party_id)
    return component_provider(providers[component_name]["provider"])


def get_component_parameters(dsl_parser, component_name, role, party_id, provider: ComponentProvider = None):
    if not provider:
        provider = get_component_provider(dsl_parser=dsl_parser,
                                          component_name=component_name,
                                          role=role,
                                          party_id=party_id)
    component_parameters_on_party = dsl_parser.parse_component_parameters(component_name,
                                                                          RuntimeConfig.COMPONENT_REGISTRY,
                                                                          provider.name,
                                                                          provider.version,
                                                                          local_role=role,
                                                                          local_party_id=party_id)
    return component_parameters_on_party


def get_component_run_info(dsl_parser, component_name, role, party_id):
    provider = get_component_provider(dsl_parser, component_name, role, party_id)
    parameters = get_component_parameters(dsl_parser, component_name, role, party_id, provider)
    return provider, parameters


def get_component_framework_interface(provider: ComponentProvider):
    return get_component_class_path(provider, "framework_interface")


def get_component_class_path(provider: ComponentProvider, class_name):
    class_path = RuntimeConfig.COMPONENT_REGISTRY["provider"].get(provider.name, {}).get(provider.version, {}).get("class_path", {}).get(class_name, None)
    if not class_path:
        class_path = RuntimeConfig.COMPONENT_REGISTRY["default_settings"]["class_path"][class_name]
    return importlib.import_module("{}.{}".format(".".join(provider.path), class_path))