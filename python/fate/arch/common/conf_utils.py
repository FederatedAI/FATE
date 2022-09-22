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
from filelock import FileLock
from importlib import import_module

from fate_arch.common import file_utils

SERVICE_CONF = "service_conf.yaml"
TRANSFER_CONF = "transfer_conf.yaml"


def conf_realpath(conf_name):
    conf_path = f"conf/{conf_name}"
    return os.path.join(file_utils.get_project_base_directory(), conf_path)


def get_base_config(key, default=None, conf_name=SERVICE_CONF) -> dict:
    local_config = {}
    local_path = conf_realpath(f'local.{conf_name}')

    if os.path.exists(local_path):
        local_config = file_utils.load_yaml_conf(local_path)
        if not isinstance(local_config, dict):
            raise ValueError(f'Invalid config file: "{local_path}".')

        if key is not None and key in local_config:
            return local_config[key]

    config_path = conf_realpath(conf_name)
    config = file_utils.load_yaml_conf(config_path)

    if not isinstance(config, dict):
        raise ValueError(f'Invalid config file: "{config_path}".')

    config.update(local_config)
    return config.get(key, default) if key is not None else config


def decrypt_database_password(password):
    encrypt_password = get_base_config("encrypt_password", False)
    encrypt_module = get_base_config("encrypt_module", False)
    private_key = get_base_config("private_key", None)

    if not password or not encrypt_password:
        return password

    if not private_key:
        raise ValueError("No private key")

    module_fun = encrypt_module.split("#")
    pwdecrypt_fun = getattr(import_module(module_fun[0]), module_fun[1])

    return pwdecrypt_fun(private_key, password)


def decrypt_database_config(database=None, passwd_key="passwd"):
    if not database:
        database = get_base_config("database", {})

    database[passwd_key] = decrypt_database_password(database[passwd_key])
    return database


def update_config(key, value, conf_name=SERVICE_CONF):
    conf_path = conf_realpath(conf_name=conf_name)
    if not os.path.isabs(conf_path):
        conf_path = os.path.join(file_utils.get_project_base_directory(), conf_path)

    with FileLock(os.path.join(os.path.dirname(conf_path), ".lock")):
        config = file_utils.load_yaml_conf(conf_path=conf_path) or {}
        config[key] = value
        file_utils.rewrite_yaml_conf(conf_path=conf_path, config=config)
