#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from contextlib import contextmanager

import yaml
from omegaconf import OmegaConf


class CrypTenConfig(object):
    """
    Configuration object used to store configurable parameters for CrypTen.

    This object acts as a nested dictionary, but can be queried using dot-notation(
    e.g. querying or setting `cfg.a.b` is equivalent to `cfg['a']['b']`).

    Users can load a CrypTen config from a file using `cfg.load_config(filepath)`.

    Users can temporarily override a config parameter using the contextmanager temp_override:

        .. code-block:: python

        cfg.a.b = outer     # sets cfg["a"]["b"] to outer value

        with cfg.temp_override("a.b", inner):
            print(cfg.a.b)  # prints inner value

        print(cfg.a.b)  # prints outer value
    """

    __DEFAULT_CONFIG_PATH = os.path.normpath(
        os.path.join(__file__, "../../../../../../../configs/default.yaml")
    )

    def __init__(self, config_file=None):
        self.load_config(config_file)

    def load_config(self, config_file):
        """Loads config from a yaml file"""
        if config_file is None:
            config_file = CrypTenConfig.__DEFAULT_CONFIG_PATH

        # Use yaml to open stream for safe load
        with open(config_file) as stream:
            config_dict = yaml.safe_load(stream)
        self.config = OmegaConf.create(config_dict)

    def set_config(self, config):
        if isinstance(config, CrypTenConfig):
            self.config = config.config
        else:
            self.config = config

    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            keys = name.split(".")
            result = getattr(self.config, keys[0])
            for key in keys[1:]:
                result = getattr(result, key)
            return result

    def __getitem__(self, name):
        return self.__getattribute__(name)

    def __setattr__(self, name, value):
        if name == "config":
            object.__setattr__(self, name, value)
        try:
            # Can only set attribute if already exists
            object.__getattribute__(self, name)
            object.__setattr__(self, name, value)
        except AttributeError:
            dotlist = [f"{name}={value}"]
            update = OmegaConf.from_dotlist(dotlist)
            self.config = OmegaConf.merge(self.config, update)

    def __setitem__(self, name, value):
        self.__setattr__(name, value)

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
