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

"""
execute with python -m fate.components --execution_id xxx --config xxx
"""
import logging

import click


@click.group()
def cli():
    pass


@cli.command()
@click.option("--execution-id", required=True, help="unique id to identify this execution")
@click.option("--config", required=False, type=click.File(), help="config path")
@click.option("--config-entrypoint", required=False, help="enctypoint to get config")
@click.option("--component", required=False, help="execution cpn")
@click.option("--stage", required=False, help="execution stage")
@click.option("--role", required=False, help="execution role")
def component(execution_id, config, config_entrypoint, component, role, stage):
    # TODO: extends parameters
    from fate.components.spec.task import TaskConfigSpec

    # parse config
    configs = {}
    load_config_from_entrypoint(configs, config_entrypoint)
    load_config_from_file(configs, config)
    load_config_from_cli(configs, execution_id, component, role, stage)
    task_config = TaskConfigSpec.parse_obj(configs)

    # install logger
    task_config.env.logger.install()
    logger = logging.getLogger(__name__)
    logger.debug("logger installed")
    logger.debug(f"task config: {task_config}")

    # init mlmd
    task_config.env.mlmd.init(task_config.execution_id)

    from fate.components.entrypoint.component import execute_component

    execute_component(task_config)


def load_config_from_cli(configs, execution_id, component, role, stage):
    configs["execution_id"] = execution_id
    if role is not None:
        configs["role"] = role
    if stage is not None:
        configs["stage"] = stage
    if component is not None:
        configs["component"] = component


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


if __name__ == "__main__":
    cli()
