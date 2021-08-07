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

import json
import pprint
import sys
from fate_flow.scheduler import dsl_parser

"""
def run(self, pipeline_dsl=None, pipeline_runtime_conf=None, dsl=None, runtime_conf=None,
        provider_detail=None, mode="train", local_role=None,
        local_party_id=None, deploy_detail=None, *args, **kwargs):
"""

dsl_path_v2 = sys.argv[1]
conf_path_v2 = sys.argv[2]
provider_path = sys.argv[3]

"""test dsl v2"""
with open(dsl_path_v2, "r") as fin:
    dsl_v2 = json.loads(fin.read())

with open(conf_path_v2, "r") as fin:
    conf_v2 = json.loads(fin.read())

with open(provider_path, "r") as fin:
    provider_detail = json.loads(fin.read())


dsl_parser_v2 = dsl_parser.DSLParserV2()
dsl_parser_v2.run(dsl=dsl_v2,
                  runtime_conf=conf_v2,
                  mode="train")

pprint.pprint(dsl_parser_v2.get_job_parameters())
print ("\n\n\n")
pprint.pprint(dsl_parser_v2.get_job_providers(provider_detail=provider_detail,
                                              local_role="arbiter",
                                              local_party_id=9999))
print ("\n\n\n")
pprint.pprint(dsl_parser_v2.get_dependency())
print ("\n\n\n")

job_providers = dsl_parser_v2.get_job_providers(provider_detail=provider_detail,
                                                local_role="arbiter",
                                                local_party_id=9999)
component_parameters = dict()
for component in job_providers.keys():
    provider_info = job_providers[component]["provider"]
    provider_name = provider_info["name"]
    provider_version = provider_info["version"]

    parameter = dsl_parser_v2.parse_component_parameters(component,
                                                         provider_detail,
                                                         provider_name,
                                                         provider_version,
                                                         local_role="arbiter",
                                                         local_party_id=9999)

    component_parameters[component] = parameter
    pprint.pprint (parameter)

pprint.pprint(dsl_parser_v2.get_dependency_with_parameters(component_parameters))
print ("\n\n\n")


print (dsl_parser_v2.get_dsl_hierarchical_structure())
print (dsl_parser_v2.get_dsl_hierarchical_structure()[0]["reader_0"].get_component_provider())
print ("\n\n\n")

pprint.pprint(dsl_parser_v2.deploy_component(["reader_0", "dataio_0"], dsl_v2))
print ("\n\n\n")


pprint.pprint(dsl_parser_v2.get_job_providers_by_conf(dsl_v2, conf_v2, provider_detail,
                                                      "guest", 9999))
print ("\n\n\n")

module_object_name_mapping = dict()
for component in job_providers.keys():
    module = dsl_v2["components"][component]["module"]
    provider_info = job_providers[component]["provider"]
    provider_name = provider_info["name"]
    provider_version = provider_info["version"]
    module_object = dsl_parser_v2.get_module_object_name(module, "guest", provider_detail,
                                                         provider_name, provider_version)
    print(f"{component} {module} {module_object}")

    module_object_name_mapping[component] = module_object


pprint.pprint(dsl_parser_v2.get_predict_dsl(dsl_v2, module_object_name_mapping))
