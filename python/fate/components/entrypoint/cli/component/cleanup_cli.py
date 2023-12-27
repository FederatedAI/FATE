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
    from fate.components.core import is_root_worker

    configs = {}
    configs = load_config_from_env(configs, env_name)
    load_config_from_file(configs, config)
    config = TaskCleanupConfigSpec.parse_obj(configs)

    try:
        if is_root_worker():
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
