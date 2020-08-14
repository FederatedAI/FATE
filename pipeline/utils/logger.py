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

import sys
from pathlib import Path

import loguru

from pipeline.backend.config import LogPath, LogFormat

LOGGER = loguru.logger
LOGGER.remove()
LOGGER.add(sys.stderr, level="INFO", colorize=True, format=LogFormat.SIMPLE)
LOGGER.add(Path(LogPath.INFO).resolve(), level="INFO", rotation="500MB", format=LogFormat.NORMAL)
LOGGER.add(Path(LogPath.DEBUG).resolve(), level="DEBUG", rotation="500MB", format=LogFormat.NORMAL)
LOGGER.add(Path(LogPath.ERROR).resolve(), level="ERROR", rotation="500MB", format=LogFormat.NORMAL, backtrace=True)
