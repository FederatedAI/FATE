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

import yaml
from pathlib import Path


__all__ = ["StatusCode", "LogPath", "LogFormat", "FlowConfig", "FateStandaloneConfig"]


with Path(__file__).parent.parent.joinpath("pipeline_config.yaml").resolve().open("r") as fin:
    __DEFAULT_CONFIG: dict = yaml.safe_load(fin)


def get_default_config():
    return __DEFAULT_CONFIG


CONSOLE_DISPLAY_LOG = get_default_config().get("console_display_log", True)
if CONSOLE_DISPLAY_LOG is None:
    CONSOLE_DISPLAY_LOG = True


class StatusCode(object):
    SUCCESS = 0
    FAIL = 1
    CANCELED = 2


class FlowConfig(object):
    conf = get_default_config().get("fate_flow", {})
    IP = conf.get("ip", None)
    PORT = conf.get("port", None)


class FateStandaloneConfig(object):
    conf = get_default_config().get("fate_standalone", {})

    default_path = Path.cwd()
    OUTPUT_DATA_DIR = conf.get("output_data_dir")
    if not OUTPUT_DATA_DIR:
        OUTPUT_DATA_DIR = default_path.joinpath("data").as_uri()
    OUTPUT_MODEL_DIR = conf.get("output_model_dir")

    if not OUTPUT_MODEL_DIR:
        OUTPUT_MODEL_DIR = default_path.joinpath("model").as_uri()
    OUTPUT_METRIC_DIR = conf.get("output_metric_dir")

    if not OUTPUT_METRIC_DIR:
        OUTPUT_METRIC_DIR = default_path.joinpath("metric").as_uri()

    JOB_DIR = conf.get("job_dir")
    if not JOB_DIR:
        JOB_DIR = default_path.joinpath("jobs").as_uri()

    OUTPUT_STATUS_DIR = conf.get("output_status_dir")
    if not OUTPUT_STATUS_DIR:
        OUTPUT_STATUS_DIR = default_path.joinpath("status").as_uri()


class LogPath(object):
    @classmethod
    def log_directory(cls):
        conf = get_default_config()
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
