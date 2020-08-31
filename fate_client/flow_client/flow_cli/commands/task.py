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
from flow_client.flow_cli.utils import cli_args
from flow_client.flow_cli.utils.cli_utils import preprocess, access_server


@click.group(short_help="Task Operations")
@click.pass_context
def task(ctx):
    """
    \b
    Provides numbers of task operational commands, including list and query.
    For more details, please check out the help text.
    """
    pass


@task.command("list", short_help="List Task Command")
@cli_args.LIMIT
@click.pass_context
def list_task(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        List Task description

    \b
    - USAGE:
        flow task list
        flow task list -l 25
    """
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, 'job/list/task', config_data)


@task.command("query", short_help="Query Task Command")
@cli_args.JOBID
@cli_args.ROLE
@cli_args.PARTYID
@cli_args.COMPONENT_NAME
@cli_args.STATUS
@click.pass_context
def query(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        Query Task Command.

    \b
    - USAGE:
        flow task query -j $JOB_ID -p 9999 -r guest
        flow task query -cpn hetero_feature_binning_0 -s success
    """
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, 'job/task/query', config_data)
