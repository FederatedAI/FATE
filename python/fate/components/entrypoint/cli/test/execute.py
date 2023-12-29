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
import sys

import click


@click.command()
@click.option("--config-path", type=click.Path(exists=True), required=True)
@click.option("--properties", "-p", multiple=True, help="properties config")
def execute(config_path, properties):
    """
    execute component from existing config file and data path, for debug purpose
    Args:
        config_path:

    Returns:

    """
    sys.argv = [__name__, "--config", f"{config_path}", "--debug"] + [f"--properties={p}" for p in properties]
    from fate.components.entrypoint.cli.component.execute_cli import execute

    execute()


if __name__ == "__main__":
    execute()
