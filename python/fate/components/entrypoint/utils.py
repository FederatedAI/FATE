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


def load_properties(properties) -> dict:
    properties_dict = {}
    for property_item in properties:
        k, v = property_item.split("=")
        k = k.strip()
        v = v.strip()
        properties_dict[k] = v
    return properties_dict


def load_properties_from_env(env_filter_prefix):
    import os

    properties_dict = {}
    if env_filter_prefix:
        env_prefix_size = len(env_filter_prefix)
        for k, v in os.environ.items():
            if k.startswith(env_filter_prefix):
                property_key = k[env_prefix_size:]
                if property_key:
                    properties_dict[property_key] = v
    return properties_dict


def load_config_from_properties(configs, properties_dict):
    for k, v in properties_dict.items():
        lens_and_setter = configs, None

        def _setter(d, k):
            def _set(v):
                d[k] = v

            return _set

        for s in k.split("."):
            lens, _ = lens_and_setter
            if not s.endswith("]"):
                if lens.get(s) is None:
                    lens[s] = {}
                lens_and_setter = lens[s], _setter(lens, s)
            else:
                name, index = s.rstrip("]").split("[")
                index = int(index)
                if lens.get(name) is None:
                    lens[name] = []
                lens = lens[name]
                if (short_size := index + 1 - len(lens)) > 0:
                    lens.extend([None] * short_size)
                    lens[index] = {}
                lens_and_setter = lens[index], _setter(lens, index)
        _, setter = lens_and_setter
        if setter is not None:
            setter(v)


def load_config_from_file(configs, config_file):
    from ruamel import yaml

    if config_file is not None:
        configs.update(yaml.safe_load(config_file))
    return configs


def load_config_from_entrypoint(configs, config_entrypoint):
    import requests

    if config_entrypoint is not None:
        try:
            resp = requests.get(config_entrypoint).json()
            configs.update(resp["config"])
        except:
            pass
    return configs


def load_config_from_env(configs, env_name):
    import os

    from ruamel import yaml

    if env_name is not None and os.environ.get(env_name):
        configs.update(yaml.safe_load(os.environ[env_name]))
    return configs
