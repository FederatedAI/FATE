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

from pathlib import Path

from pipeline.backend import get_default_config
from pipeline.constant import JobStatus

__all__ = ["JobStatus", "VERSION", "SERVER_VERSION", "TIME_QUERY_FREQS", "Role", "StatusCode",
           "LogPath", "LogFormat", "IODataType", "FlowConfig"]

VERSION = 2
SERVER_VERSION = "v1"
TIME_QUERY_FREQS = 0.5


CONSOLE_DISPLAY_LOG = get_default_config().get("console_display_log", True)
if CONSOLE_DISPLAY_LOG is None:
    CONSOLE_DISPLAY_LOG = True


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
    CANCELED = 2


class IODataType:
    SINGLE = "data"
    TRAIN = "train_data"
    VALIDATE = "validate_data"
    TEST = "test_data"


class FlowConfig(object):
    conf = get_default_config()
    IP = conf.get("ip", None)
    if IP is None:
        raise ValueError(f"IP not configured. "
                         f"Please use command line tool pipeline init to set Flow server IP.")
    PORT = conf.get("port", None)
    if PORT is None:
        raise ValueError(f"PORT not configured. "
                         f"Please use command line tool pipeline init to set Flow server port")

    APP_KEY = conf.get("app_key", None)
    SECRET_KEY = conf.get("secret_key", None)


class LogPath(object):
    @classmethod
    def log_directory(cls):
        conf = get_default_config()
        # log_directory = os.environ.get("FATE_PIPELINE_LOG", "")
        log_directory = conf.get("log_directory")
        if log_directory:
            log_directory = Path(log_directory).resolve()
        else:
            log_directory = Path(__file__).parent.parent.joinpath("logs")
        try:
            log_directory.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"can't create log directory for pipeline: {log_directory}") from e
        if not Path(log_directory).resolve().is_dir():
            raise NotADirectoryError(f"provided log directory {log_directory} is not a directory.")
        return log_directory

    DEBUG = 'DEBUG.log'
    INFO = 'INFO.log'
    ERROR = 'ERROR.log'


class LogFormat(object):
    SIMPLE = '<green>[{time:HH:mm:ss}]</green><level>{message}</level>'
    NORMAL = '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | ' \
             '<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>'


class SystemSetting(object):
    @classmethod
    def system_setting(cls):
        conf = get_default_config()
        system_setting = conf.get("system_setting", {})
        # system_role = system_setting.get("role", None)
        return system_setting
