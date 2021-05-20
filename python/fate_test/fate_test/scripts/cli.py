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

import click
from fate_test.scripts._options import SharedOptions
from fate_test.scripts.benchmark_cli import run_benchmark
from fate_test.scripts.config_cli import config_group
from fate_test.scripts.data_cli import data_group
from fate_test.scripts.testsuite_cli import run_suite
from fate_test.scripts.performance_cli import run_task
from fate_test.scripts.flow_test_cli import flow_group

commands = {
    "config": config_group,
    "suite": run_suite,
    "performance": run_task,
    "benchmark-quality": run_benchmark,
    "data": data_group,
    "flow-test": flow_group
}

commands_alias = {
    "bq": "benchmark-quality",
    "perf": "performance"
}


class MultiCLI(click.MultiCommand):

    def list_commands(self, ctx):
        return list(commands)

    def get_command(self, ctx, name):
        if name not in commands and name in commands_alias:
            name = commands_alias[name]
        if name not in commands:
            ctx.fail("No such command '{}'.".format(name))
        return commands[name]


@click.command(cls=MultiCLI, help="A collection of useful tools to running FATE's test.",
               context_settings=dict(help_option_names=["-h", "--help"]))
@SharedOptions.get_shared_options()
@click.pass_context
def cli(ctx, **kwargs):
    ctx.ensure_object(SharedOptions)
    ctx.obj.update(**kwargs)


if __name__ == '__main__':
    cli(obj=SharedOptions())
