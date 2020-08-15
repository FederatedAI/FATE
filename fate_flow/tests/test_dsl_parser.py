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
import os
from fate_flow.scheduler.dsl_parser import DSLParser
from fate_arch.common import file_utils

dsl_parser = DSLParser()
default_runtime_conf_path = os.path.join(file_utils.get_project_base_directory(),
                                         *['federatedml', 'conf', 'default_runtime_conf'])
setting_conf_path = os.path.join(file_utils.get_project_base_directory(), *['federatedml', 'conf', 'setting_conf'])
dsl = file_utils.load_json_conf("fate_flow/examples/test_hetero_lr_job_dsl.json")
runtime_conf = file_utils.load_json_conf("fate_flow/examples/test_hetero_lr_job_conf.json")
dsl_parser.run(dsl=dsl,
               runtime_conf=runtime_conf,
               default_runtime_conf_prefix=default_runtime_conf_path,
               setting_conf_prefix=setting_conf_path,
               mode="train")
tasksets = dsl_parser.get_dsl_hierarchical_structure()
print(tasksets)
print(type(tasksets))
