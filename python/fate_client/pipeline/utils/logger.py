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

from pipeline.backend.config import LogPath, LogFormat, CONSOLE_DISPLAY_LOG

RUNTIME_LOG = "runtime"

info_log_path = os.path.join(LogPath.log_directory(), LogPath.INFO)
debug_log_path = os.path.join(LogPath.log_directory(), LogPath.DEBUG)
error_log_path = os.path.join(LogPath.log_directory(), LogPath.ERROR)


def runtime_log_only(record):
    log_type = record["extra"].get("log_type", "")
    return log_type == RUNTIME_LOG


LOGGER = loguru.logger

LOGGER.remove()
LOGGER.configure(extra={"format": LogFormat.NORMAL})
if CONSOLE_DISPLAY_LOG:
    console_handler = LOGGER.add(sys.stderr, level="INFO", colorize=True,
                                 filter=runtime_log_only)
LOGGER.add(Path(info_log_path).resolve(), level="INFO", rotation="500MB",
           colorize=True, filter=runtime_log_only)
LOGGER.add(Path(debug_log_path).resolve(), level="DEBUG", rotation="500MB", colorize=True,
           filter=runtime_log_only)
LOGGER.add(Path(error_log_path).resolve(), level="ERROR", rotation="500MB", colorize=True,
           backtrace=True, filter=runtime_log_only)
LOGGER = LOGGER.bind(log_type=RUNTIME_LOG)


def disable_console_log():
    """
    disable logging to stderr, for silent mode
    Returns
    -------

    """
    try:
        LOGGER.remove(console_handler)
    except:
        pass


def enable_console_log():
    disable_console_log()
    global console_handler
    console_handler = LOGGER.add(sys.stderr, level="INFO", colorize=True,
                                 filter=runtime_log_only)
