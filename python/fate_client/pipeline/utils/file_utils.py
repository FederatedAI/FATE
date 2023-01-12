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
import json
import typing
from pathlib import Path
import yaml
from .uri_tools import parse_uri


def construct_local_file(filepath: typing.Union[Path, str], *suffixes) -> "Path":
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    for suf in suffixes:
        filepath = filepath.joinpath(suf)

    filepath.parent.mkdir(parents=True, exist_ok=True)

    return filepath


def construct_local_dir(filepath: typing.Union[Path, str], *suffixes) -> "Path":
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    for suf in suffixes:
        filepath = filepath.joinpath(suf)

    filepath.mkdir(parents=True, exist_ok=True)

    return filepath


def write_json_file(path: str, buffer: dict):
    path = parse_uri(path).path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fout:
        fout.write(json.dumps(buffer, indent=2))
        fout.flush()


def write_yaml_file(path: str, buffer: dict):
    path = parse_uri(path).path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fout:
        fout.write(yaml.dump(buffer, indent=2))
        fout.flush()


def load_yaml_file(path: str):
    with open(path, "r") as fin:
        buf = fin.read()
        return yaml.safe_load(buf)
