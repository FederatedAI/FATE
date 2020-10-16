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

import json
import typing
from pathlib import Path
from ruamel import yaml


def merge_dict(dict1, dict2):
    merge_ret = {}
    keyset = dict1.keys() | dict2.keys()
    for key in keyset:
        if key in dict1 and key in dict2:
            val1 = dict1.get(key)
            val2 = dict2.get(key)
            assert type(val1).__name__ == type(val2).__name__
            if isinstance(val1, dict):
                merge_ret[key] = merge_dict(val1, val2)
            else:
                merge_ret[key] = val2
        elif key in dict1:
            merge_ret[key] = dict1.get(key)
        else:
            merge_ret[key] = dict2.get(key)

    return merge_ret


def extract_explicit_parameter(func):
    def wrapper(*args, **kwargs):
        explict_kwargs = {"explict_parameters": kwargs}

        return func(*args, **explict_kwargs)

    return wrapper


def load_job_config(path):
    config = JobConfig.load(path)
    return config


class Parties(object):
    def __init__(self, parties):
        self.host = parties.get("host", None)
        self.guest = parties.get("guest", None)
        self.arbiter = parties.get("arbiter", None)


class JobConfig(object):
    def __init__(self, config):
        self.parties = Parties(config.get("parties", {}))
        self.backend = config.get("backend", 0)
        self.work_mode = config.get("work_mode", 0)
        self.data_base = config.get("data_base", 0)

    @staticmethod
    def load(path: typing.Union[str, Path]):
        conf = JobConfig.load_from_file(path)
        return JobConfig(conf)

    @staticmethod
    def load_from_file(path: typing.Union[str, Path]):
        """
        Loads conf content from json or yaml file. Used to read in parameter configuration
        Parameters
        ----------
        path: str, path to conf file, should be absolute path

        Returns
        -------
        dict, parameter configuration in dictionary format

        """
        if isinstance(path, str):
            path = Path(path)
        config = {}
        if path is not None:
            file_type = path.suffix
            with path.open("r") as f:
                if file_type == ".yaml":
                    config.update(yaml.safe_load(f))
                elif file_type == ".json":
                    config.update(json.load(f))
                else:
                    raise ValueError(f"Cannot load conf from file type {file_type}")
        return config
