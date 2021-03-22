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
from flow_client.flow_cli.utils.cli_utils import access_server


@click.group(short_help="Queue Operations")
@click.pass_context
def queue(ctx):
    """
    \b
    Provides a queue operational command, which is 'clean'.
    For more details, please check out the help text.
    """
    pass


@queue.command("clean", short_help="Clean Queue Command")
@click.pass_context
def clean(ctx):
    """
    \b
    - DESCRIPTION:
        Queue Clean Command

    \b
    - USAGE:
        flow queue clean
    """
    access_server('post', ctx, "job/clean/queue")
