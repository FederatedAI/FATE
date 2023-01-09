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
import click


@click.group()
def component():
    """
    Manipulate components: execute, list, generate describe file
    """


@component.command()
@click.option("--process-tag", required=True, help="unique id to identify this execution process")
@click.option("--config", required=False, type=click.File(), help="config path")
@click.option("--config-entrypoint", required=False, help="enctypoint to get config")
@click.option("--properties", "-p", multiple=True, help="properties config")
@click.option("--env-prefix", "-e", type=str, default="runtime.component.", help="prefix for env config")
def execute(process_tag, config, config_entrypoint, properties, env_prefix):
    "execute component"
    import logging

    from fate.components.spec.task import TaskConfigSpec

    # parse properties
    properties_items = {}
    properties_items.update(load_properties(properties))
    properties_items.update(load_properties_from_env(env_prefix))

    # parse config
    configs = {}
    load_config_from_entrypoint(configs, config_entrypoint)
    load_config_from_file(configs, config)
    load_config_from_properties(configs, properties_items)

    task_config = TaskConfigSpec.parse_obj(configs)

    # install logger
    task_config.conf.logger.install()
    logger = logging.getLogger(__name__)
    logger.debug("logger installed")
    logger.debug(f"task config: {task_config}")

    from fate.components.entrypoint.component import execute_component

    execute_component(task_config)


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
                print("in", lens)
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


@component.command()
@click.option("--name", required=True, help="name of component")
@click.option("--save", type=click.File(mode="w", lazy=True), help="save desc output to specified file in yaml format")
def desc(name, save):
    "generate component describe config"
    from fate.components.loader.component import load_component

    cpn = load_component(name)
    if save:
        cpn.dump_yaml(save)
    else:
        print(cpn.dump_yaml())


@component.command()
@click.option("--save", type=click.File(mode="w", lazy=True), help="save desc output to specified file in yaml format")
def task_schema(save):
    "generate component task config json schema"
    from fate.components.spec.task import TaskConfigSpec

    if save:
        save.write(TaskConfigSpec.schema_json())
    else:
        print(TaskConfigSpec.schema_json())


@component.command()
@click.option("--save", type=click.File(mode="w", lazy=True), help="save list output to specified file in json format")
def list(save):
    "list all components"
    from fate.components.loader.component import list_components

    if save:
        import json

        json.dump(list_components(), save)
    else:
        print(list_components())


@component.command()
@click.option("--db", required=True, type=str, help="mlmd db")
@click.option("--taskid", required=True, type=str, help="taskid")
def set_mlmd_finish(db, taskid):
    from fate.arch.context._mlmd import MachineLearningMetadata

    mlmd = MachineLearningMetadata(metadata={"filename_uri": db})
    mlmd.set_task_safe_terminate_flag(taskid)
