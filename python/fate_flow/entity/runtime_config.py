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

from fate_arch.common.versions import get_versions


class RuntimeConfig(object):
    WORK_MODE = None
    COMPUTING_ENGINE = None
    FEDERATION_ENGINE = None
    FEDERATED_MODE = None

    JOB_QUEUE = None
    USE_LOCAL_DATABASE = False
    HTTP_PORT = None
    JOB_SERVER_HOST = None
    JOB_SERVER_VIP = None
    IS_SERVER = False
    PROCESS_ROLE = None
    ENV = dict()

    @classmethod
    def init_config(cls, **kwargs):
        for k, v in kwargs.items():
            if hasattr(RuntimeConfig, k):
                setattr(RuntimeConfig, k, v)

    @classmethod
    def init_env(cls):
        RuntimeConfig.ENV.update(get_versions())

    @classmethod
    def get_env(cls, key):
        return RuntimeConfig.ENV.get(key, None)

    @classmethod
    def set_process_role(cls, process_role: PROCESS_ROLE):
        RuntimeConfig.PROCESS_ROLE = process_role
