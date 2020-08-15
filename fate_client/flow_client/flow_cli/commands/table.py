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


@click.group(short_help="Table Operations")
@click.pass_context
def table(ctx):
    """
    \b
    Provides numbers of table operational commands, including info and delete.
    For more details, please check out the help text.
    """
    pass


@table.command("info", short_help="Query Table Command")
@cli_args.NAMESPACE_REQUIRED
@cli_args.TABLE_NAME_REQUIRED
@click.pass_context
def info(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        Query Table Information.

    \b
    - USAGE:
        flow table info -n $NAMESPACE -t $TABLE_NAME
    """
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, 'table/table_info', config_data)


@table.command("delete", short_help="Delete Table Command")
@cli_args.NAMESPACE
@cli_args.TABLE_NAME
@cli_args.JOBID
@cli_args.ROLE
@cli_args.PARTYID
@cli_args.COMPONENT_NAME
@click.pass_context
def delete(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        Delete A Specified Table.

    \b
    - USAGE:
        flow table delete -n $NAMESPACE -t $TABLE_NAME
        flow table delete -j $JOB_ID -r guest -p 9999
    """
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, 'table/delete', config_data)
