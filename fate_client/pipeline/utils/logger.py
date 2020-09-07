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
import sys
from pathlib import Path

import loguru

from pipeline.backend.config import LogPath, LogFormat

RUNTIME_LOG = "runtime"

def runtime_log_only(record):
    log_type = record["extra"].get("log_type", "")
    return log_type == RUNTIME_LOG


info_log_path = os.path.join(LogPath.log_directory(), LogPath.INFO)
debug_log_path = os.path.join(LogPath.log_directory(), LogPath.DEBUG)
error_log_path = os.path.join(LogPath.log_directory(), LogPath.ERROR)

LOGGER = loguru.logger
# LOGGER.remove()
# LOGGER.add(sys.stderr, level="INFO", colorize=True, format=LogFormat.SIMPLE)
#LOGGER.add(lambda msg: sys.stdout.write(msg), level="INFO", colorize=True,
#           format=LogFormat.SIMPLE, filter=runtime_log_only)
LOGGER.configure(handlers=[{"sink": sys.stderr, "level": "INFO"}])
LOGGER.add(Path(info_log_path).resolve(), level="INFO", rotation="500MB",
           colorize=True, format=LogFormat.NORMAL, filter=runtime_log_only)
LOGGER.add(Path(debug_log_path).resolve(), level="DEBUG", rotation="500MB", colorize=True,
           format=LogFormat.NORMAL, filter=runtime_log_only)
LOGGER.add(Path(error_log_path).resolve(), level="ERROR", rotation="500MB", colorize=True,
           format=LogFormat.NORMAL, backtrace=True, filter=runtime_log_only)
LOGGER = LOGGER.bind(log_type=RUNTIME_LOG)
