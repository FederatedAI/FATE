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


@click.group(short_help="Resource Manager")
@click.pass_context
def resource(ctx):
    """
    \b
    Provides numbers of resource operational commands, including query and return.
    For more details, please check out the help text.
    """
    pass


@resource.command("query", short_help="Query Resource Command")
@click.pass_context
def query(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        Query Resource Information.

    \b
    - USAGE:
        flow resource query
    """
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, 'resource/query', config_data)


@resource.command("return", short_help="Return Job Resource Command")
@cli_args.JOBID
@click.pass_context
def resource_return(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        Return Job Resource Command

    \b
    - USAGE:
        flow resource return -j $JobId
    """
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, 'resource/return', config_data)
