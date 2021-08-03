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

dsl_path_v1 = sys.argv[1]
conf_path_v1 = sys.argv[2]
provider_path = sys.argv[3]

"""test dsl v2"""
with open(dsl_path_v1, "r") as fin:
    dsl_v1 = json.loads(fin.read())

with open(conf_path_v1, "r") as fin:
    conf_v1 = json.loads(fin.read())

with open(provider_path, "r") as fin:
    provider_detail = json.loads(fin.read())


dsl_parser_v1 = dsl_parser.DSLParser()
dsl_parser_v1.run(dsl=dsl_v1,
                  runtime_conf=conf_v1,
                  mode="train")

pprint.pprint(dsl_parser_v1.get_job_parameters())
print ("\n\n\n")
pprint.pprint(dsl_parser_v1.get_job_providers(provider_detail=provider_detail,
                                              local_role="guest",
                                              local_party_id=10000))
print ("\n\n\n")
pprint.pprint(dsl_parser_v1.get_dependency())
print ("\n\n\n")

job_providers = dsl_parser_v1.get_job_providers(provider_detail=provider_detail,
                                                local_role="guest",
                                                local_party_id=10000)
component_parameters = dict()
deploy_detail = dict()
for component in job_providers.keys():
    provider_info = job_providers[component]["provider"]
    provider_name = provider_info["name"]
    provider_version = provider_info["version"]

    parameter = dsl_parser_v1.parse_component_parameters(component,
                                                         provider_detail,
                                                         provider_name,
                                                         provider_version,
                                                         local_role="guest",
                                                         local_party_id=10000)

    component_parameters[component] = parameter
    deploy_detail[component] = dsl_parser_v1.get_component_need_deploy_info(component, provider_detail, job_providers)
    # pprint.pprint(parameter)


pprint.pprint(deploy_detail)
pprint.pprint(dsl_parser_v1.get_dependency_with_parameters(component_parameters))
print ("\n\n\n")

print (dsl_parser_v1.get_dsl_hierarchical_structure())
print (dsl_parser_v1.get_dsl_hierarchical_structure()[0]["dataio_0"].get_component_provider())

predict_dsl = dsl_parser_v1.generate_predict_dsl(deploy_detail)
pprint.pprint(predict_dsl)
print ("\n\n\n")
pprint.pprint(dsl_parser_v1.get_predict_dsl(component_parameters=component_parameters))

