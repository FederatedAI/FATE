# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
from contextlib import contextmanager

from ruamel import yaml
from omegaconf import OmegaConf


class Config(object):
    """
    Configuration object used to store configurable parameters for FATE.

    This class is a modified copy of CrypTenConfig
    in https://github.com/facebookresearch/CrypTen/blob/main/crypten/config/config.py
    """

    __DEFAULT_CONFIG_PATH = os.path.normpath(os.path.join(__file__, "../../../../../configs/default.yaml"))

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

    @property
    def debug(self):
        return self.config.debug

    @property
    def encoder(self):
        return self.config.encoder

    @property
    def functions(self):
        return self.config.functions

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


if __name__ == "__main__":
    config = Config()
    print(config.mpc.provider)
