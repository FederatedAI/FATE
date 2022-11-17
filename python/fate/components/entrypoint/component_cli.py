import click


@click.group()
def component():
    """
    Manipulate components: execute, list, generate describe file
    """


@component.command()
@click.option("--execution-id", required=True, help="unique id to identify this execution")
@click.option("--config", required=False, type=click.File(), help="config path")
@click.option("--config-entrypoint", required=False, help="enctypoint to get config")
@click.option("--component", required=False, help="execution cpn")
@click.option("--stage", required=False, help="execution stage")
@click.option("--role", required=False, help="execution role")
def execute(execution_id, config, config_entrypoint, component, role, stage):
    "execute component"
    # TODO: extends parameters
    import logging

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


@component.command()
@click.option("--name", required=True, help="name of component")
@click.option("--save", type=click.File(mode="w", lazy=True), help="save desc output to specified file in yaml format")
def desc(name, save):
    "generate component describe config"
    from fate.components.loader import load_component

    cpn = load_component(name)
    if save:
        cpn.dump_yaml(save)
    else:
        print(cpn.dump_yaml())


@component.command()
@click.option("--save", type=click.File(mode="w", lazy=True), help="save list output to specified file in json format")
def list(save):
    "list all components"
    from fate.components.loader import list_components

    if save:
        import json

        json.dump(list_components(), save)
    else:
        print(list_components())
