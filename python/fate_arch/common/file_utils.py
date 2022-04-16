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
import os

from cachetools import LRUCache, cached
from ruamel import yaml

PROJECT_BASE = os.getenv("FATE_PROJECT_BASE") or os.getenv("FATE_DEPLOY_BASE")
FATE_BASE = os.getenv("FATE_BASE")
READTHEDOC = os.getenv("READTHEDOC")


def get_project_base_directory(*args):
    global PROJECT_BASE
    global READTHEDOC
    if PROJECT_BASE is None:
        PROJECT_BASE = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                os.pardir,
                os.pardir,
                os.pardir,
            )
        )
        if READTHEDOC is None:
            PROJECT_BASE = os.path.abspath(
                os.path.join(
                    PROJECT_BASE,
                    os.pardir,
                )
            )
    if args:
        return os.path.join(PROJECT_BASE, *args)
    return PROJECT_BASE


def get_fate_directory(*args):
    global FATE_BASE
    if FATE_BASE is None:
        FATE_BASE = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                os.pardir,
                os.pardir,
                os.pardir,
            )
        )
    if args:
        return os.path.join(FATE_BASE, *args)
    return FATE_BASE


def get_fate_python_directory(*args):
    return get_fate_directory("python", *args)


def get_federatedml_setting_conf_directory():
    return os.path.join(get_fate_python_directory(), 'federatedml', 'conf', 'setting_conf')


@cached(cache=LRUCache(maxsize=10))
def load_json_conf(conf_path):
    if os.path.isabs(conf_path):
        json_conf_path = conf_path
    else:
        json_conf_path = os.path.join(get_project_base_directory(), conf_path)
    try:
        with open(json_conf_path) as f:
            return json.load(f)
    except BaseException:
        raise EnvironmentError(
            "loading json file config from '{}' failed!".format(json_conf_path)
        )


def dump_json_conf(config_data, conf_path):
    if os.path.isabs(conf_path):
        json_conf_path = conf_path
    else:
        json_conf_path = os.path.join(get_project_base_directory(), conf_path)
    try:
        with open(json_conf_path, "w") as f:
            json.dump(config_data, f, indent=4)
    except BaseException:
        raise EnvironmentError(
            "loading json file config from '{}' failed!".format(json_conf_path)
        )


def load_json_conf_real_time(conf_path):
    if os.path.isabs(conf_path):
        json_conf_path = conf_path
    else:
        json_conf_path = os.path.join(get_project_base_directory(), conf_path)
    try:
        with open(json_conf_path) as f:
            return json.load(f)
    except BaseException:
        raise EnvironmentError(
            "loading json file config from '{}' failed!".format(json_conf_path)
        )


def load_yaml_conf(conf_path):
    if not os.path.isabs(conf_path):
        conf_path = os.path.join(get_project_base_directory(), conf_path)
    try:
        with open(conf_path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise EnvironmentError(
            "loading yaml file config from {} failed:".format(conf_path), e
        )


def rewrite_yaml_conf(conf_path, config):
    if not os.path.isabs(conf_path):
        conf_path = os.path.join(get_project_base_directory(), conf_path)
    try:
        with open(conf_path, "w") as f:
            yaml.dump(config, f, Dumper=yaml.RoundTripDumper)
    except Exception as e:
        raise EnvironmentError(
            "rewrite yaml file config {} failed:".format(conf_path), e
        )


def rewrite_json_file(filepath, json_data):
    with open(filepath, "w") as f:
        json.dump(json_data, f, indent=4, separators=(",", ": "))
    f.close()
