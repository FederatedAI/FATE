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
# -*- coding: utf-8 -*-
from fate_arch.computing import ComputingEngine
from fate_arch.common.conf_utils import get_base_config
from fate_flow.settings import FATEFLOW_SERVICE_NAME

# Resource
TOTAL_CORES_OVERWEIGHT_PERCENT = 1  # 1 means no overweight
TOTAL_MEMORY_OVERWEIGHT_PERCENT = 1  # 1 means no overweight
TASK_PARALLELISM = 1
TASK_CORES = 4
TASK_MEMORY = 0  # mb
MAX_CORES_PERCENT_PER_JOB = 1  # 1 means total
IGNORE_RESOURCE_ROLES = {"arbiter"}
SUPPORT_IGNORE_RESOURCE_ENGINES = {
    ComputingEngine.EGGROLL, ComputingEngine.STANDALONE
}


# Scheduling
DEFAULT_REMOTE_REQUEST_TIMEOUT = 30 * 1000  # ms
DEFAULT_FEDERATED_COMMAND_TRYS = 3
JOB_DEFAULT_TIMEOUT = 3 * 24 * 60 * 60
END_STATUS_JOB_SCHEDULING_TIME_LIMIT = 5 * 60 * 1000  # ms
END_STATUS_JOB_SCHEDULING_UPDATES = 1
FEDERATED_STATUS_COLLECT_TYPE = get_base_config(FATEFLOW_SERVICE_NAME, {}).get("default_federated_status_collect_type", "PUSH")
AUTO_RETRIES = 0
AUTO_RETRY_DELAY = 1  #seconds