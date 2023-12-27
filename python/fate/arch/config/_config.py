# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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


import os
from contextlib import contextmanager

from omegaconf import OmegaConf
from ruamel import yaml


class Config(object):
    """
    Configuration object used to store configurable parameters for FATE.

    This class is a modified copy of CrypTenConfig
    in https://github.com/facebookresearch/CrypTen/blob/main/crypten/config/config.py
    """

    __DEFAULT_CONFIG_PATH = os.path.normpath(os.path.join(__file__, os.path.pardir, "default.yaml"))

    def __init__(self, config_file=None):
        if config_file is None:
            config_file = Config.__DEFAULT_CONFIG_PATH

        # Use yaml to open stream for safe load
        with open(config_file) as stream:
            config_dict = yaml.safe_load(stream)
        self.config = OmegaConf.create(config_dict)

    @property
    def mpc(self):
        return self.config.mpc

    @contextmanager
    def temp_override(self, override_dict):
        old_config = self.config
        try:
            dotlist = [f"{k}={v}" for k, v in override_dict.items()]
            update = OmegaConf.from_dotlist(dotlist)
            self.config = OmegaConf.merge(self.config, update)
            yield
        finally:
            self.config = old_config

    def get_option(self, options, key, default=...):
        if key in options:
            return options[key]
        elif self.config.get(key, None) is not None:
            return self.config[key]
        elif default is ...:
            raise ValueError(f"{key} not in {options} or {self.config}")
        else:
            return default

    @property
    def federation(self):
        return self.config.federation

    @property
    def safety(self):
        return self.config.safety

    @property
    def components(self):
        return self.config.components
