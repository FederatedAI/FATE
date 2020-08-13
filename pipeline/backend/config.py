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

VERSION = 2
TIME_QUERY_FREQS = 0.01


class Role(object):
    LOCAL = "local"
    GUEST = "guest"
    HOST = "host"
    ARBITER = "arbiter"

    @classmethod
    def support_roles(cls):
        roles = set()
        for role_key, role in cls.__dict__.items():
            if role_key.startswith("__") and isinstance(role_key, str):
                continue

            roles.add(role)

        return roles


class StatusCode(object):
    SUCCESS = 0
    FAIL = 1
    Cancel = 2


class LogPath(object):
    DEBUG = "pipeline/logs/DEBUG.log"
    INFO = "pipeline/logs/INFO.log"
    ERROR = "pipeline/logs/ERROR.log"


class LogFormat(object):
    SIMPLE = '<green>[{time:HH:mm:ss}]</green><level>{message}</level>'
    NORMAL = '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | ' \
                 '<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>'
