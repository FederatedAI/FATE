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
from arch.api.utils import file_utils


# TODO: add config cache
def get_base_config(key, default=None):
    base_config = file_utils.load_yaml_conf(os.path.join(file_utils.get_project_base_directory(), "arch/conf/base_conf.yaml")) or dict()
    return base_config.get(key, default)
