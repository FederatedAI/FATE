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

from ruamel import yaml

with Path(__file__).parent.parent.joinpath("config.yaml").resolve().open("r") as fin:
    __DEFAULT_CONFIG: dict = yaml.safe_load(fin)


def set_default_config(ip: str, port: int, log_directory: str, console_display_log: bool):
    global __DEFAULT_CONFIG
    __DEFAULT_CONFIG.update(dict(ip=ip, port=port, log_directory=log_directory,
                                 console_display_log=console_display_log))


def get_default_config():
    return __DEFAULT_CONFIG
