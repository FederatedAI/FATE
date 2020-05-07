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
import dotenv

from arch.api.utils.core_utils import get_lan_ip
from arch.api.utils.file_utils import get_project_base_directory
from fate_flow.entity.constant_config import ProcessRole


class RuntimeConfig(object):
    WORK_MODE = None
    BACKEND = None
    JOB_QUEUE = None
    USE_LOCAL_DATABASE = False
    HTTP_PORT = None
    JOB_SERVER_HOST = None
    IS_SERVER = False
    PROCESS_ROLE = None
    ENV = dict()

    @staticmethod
    def init_config(**kwargs):
        for k, v in kwargs.items():
            if hasattr(RuntimeConfig, k):
                setattr(RuntimeConfig, k, v)
                if k == 'HTTP_PORT':
                    setattr(RuntimeConfig, 'JOB_SERVER_HOST', "{}:{}".format(get_lan_ip(), RuntimeConfig.HTTP_PORT))

    @staticmethod
    def init_env():
        RuntimeConfig.ENV.update(dotenv.dotenv_values(dotenv_path=os.path.join(get_project_base_directory(), ".env")))

    @staticmethod
    def get_env(key):
        return RuntimeConfig.ENV.get(key, None)

    @staticmethod
    def set_process_role(process_role: PROCESS_ROLE):
        RuntimeConfig.PROCESS_ROLE = process_role

