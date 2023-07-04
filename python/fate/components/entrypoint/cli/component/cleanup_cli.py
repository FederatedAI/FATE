import click


@click.command()
@click.option("--process-tag", required=False, help="unique id to identify this execution process")
@click.option("--config", required=False, type=click.File(), help="config path")
@click.option("--env-name", required=False, type=str, help="env name for config")
def cleanup(process_tag, config, env_name):
    """cleanup"""
    import traceback

    from fate.arch import Context
    from fate.components.core import load_computing, load_federation
    from fate.components.core.spec.task import TaskCleanupConfigSpec
    from fate.components.entrypoint.utils import (
        load_config_from_env,
        load_config_from_file,
    )

    configs = {}
    configs = load_config_from_env(configs, env_name)
    load_config_from_file(configs, config)
    config = TaskCleanupConfigSpec.parse_obj(configs)

    try:
        print("start cleanup")
        computing = load_computing(config.computing)
        federation = load_federation(config.federation, computing)
        ctx = Context(
            computing=computing,
            federation=federation,
        )
        ctx.destroy()
        print("cleanup done")
    except Exception as e:
        traceback.print_exc()
        raise e


if __name__ == "__main__":
    cleanup()
