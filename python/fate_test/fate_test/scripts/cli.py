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
import time

import click
from fate_test._config import default_config, parse_config
from fate_test.scripts._utils import _set_namespace
from fate_test.scripts.benchmark_cli import run_benchmark
from fate_test.scripts.config_cli import config_group
from fate_test.scripts.teetsuite_cli import run_suite
from fate_test.scripts.data_cli import data_group

commands = {
    "config": config_group,
    "suite": run_suite,
    "benchmark-quality": run_benchmark,
    "data": data_group
}


class MultiCLI(click.MultiCommand):
    def list_commands(self, ctx):
        return list(commands)

    def get_command(self, ctx, name):
        return commands[name]


@click.command(cls=MultiCLI, help="A collection of useful tools to running FATE's test.")
@click.option('-c', '--config', default=default_config().__str__(), type=click.Path(exists=True),
              help=f"Manual specify config file")
@click.option('-n', '--namespace', default=time.strftime('%Y%m%d%H%M%S'), type=str,
              help=f"Manual specify fate_test namespace")
@click.option('--namespace-mangling', type=bool, is_flag=True, default=False,
              help="mangling data namespace")
@click.pass_context
def cli(ctx, config, namespace, namespace_mangling):
    ctx.ensure_object(dict)
    config_inst = parse_config(config)
    ctx.obj['config'] = config_inst
    ctx.obj['namespace'] = namespace
    ctx.obj['data_namespace_mangling'] = namespace_mangling
    _set_namespace(namespace_mangling, namespace)


if __name__ == '__main__':
    cli(obj={})
