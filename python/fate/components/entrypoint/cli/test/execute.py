import os
import sys

import click


@click.command()
@click.option("--config-path", type=click.Path(exists=True), required=True)
@click.option("--data-path", type=click.Path(exists=True), required=True)
@click.option("--properties", "-p", multiple=True, help="properties config")
def execute(config_path, data_path, properties):
    """
    execute component from existing config file and data path, for debug purpose
    Args:
        config_path:
        data_path:

    Returns:

    """
    os.environ["STANDALONE_DATA_PATH"] = str(data_path)
    sys.argv = [__name__, "--config", f"{config_path}", "--debug"] + [f"--properties={p}" for p in properties]
    from fate.components.entrypoint.cli.component.execute_cli import execute

    execute()


if __name__ == "__main__":
    execute()
