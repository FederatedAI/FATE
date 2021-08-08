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
from fate_arch.common import file_utils
from fate_flow.settings import FATE_FLOW_JOB_DEFAULT_CONFIG_PATH, stat_logger
from .reload_config_base import ReloadConfigBase


class JobDefaultConfig(ReloadConfigBase):
    # Resource
    total_cores_overweight_percent = None
    total_memory_overweight_percent = None
    task_parallelism = None
    task_cores = None
    task_memory = None
    max_cores_percent_per_job = None

    # scheduling
    default_remote_request_timeout = None
    default_federated_command_trys = None
    job_default_timeout = None
    end_status_job_scheduling_time_limit = None
    end_status_job_scheduling_updates = None
    auto_retries = None
    auto_retry_delay = None
    default_federated_status_collect_type = None

    @classmethod
    def load(cls):
        conf = file_utils.load_yaml_conf(FATE_FLOW_JOB_DEFAULT_CONFIG_PATH)
        if not isinstance(conf, dict):
            raise ValueError("invalid config file")

        for k, v in conf.items():
            if hasattr(cls, k):
                setattr(cls, k, v)
            else:
                stat_logger.warning(f"job default config not supported {k}")

        return cls.get_all()