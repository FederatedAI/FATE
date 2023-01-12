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
from .data_manager import get_data_manager
from .model_manager import get_model_manager
from .metric_manager import get_metric_manager
from .status_manager import get_status_manager
from .task_conf_manager import get_task_conf_manager
from .resource_manager import StandaloneResourceManager


__all__ = [
    "get_data_manager",
    "get_model_manager",
    "get_metric_manager",
    "get_status_manager",
    "get_task_conf_manager",
    "StandaloneResourceManager"
]