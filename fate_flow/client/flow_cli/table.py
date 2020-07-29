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
from fate_flow.utils import detect_utils, cli_args
from fate_flow.utils.cli_utils import preprocess, access_server


@click.group(short_help="Table Operations")
@click.pass_context
def table(ctx):
    """
    \b
    Provides numbers of table operational commands, including info and delete.
    For more details, please check out the help text.
    """
    pass


@table.command(short_help="Query Table Command")
# @click.argument('namespace', metavar='<NAMESPACE>')
# @click.argument('table_name', metavar='<TABLE_NAME>')
@cli_args.NAMESPACE_REQUIRED
@cli_args.TABLE_NAME_REQUIRED
@click.pass_context
def info(ctx, **kwargs):
    """
    - COMMAND DESCRIPTION:

    Query Table Information.

    - REQUIRED ARGUMENTS:

    \b
    <NAMESPACE> : Namespace
    <TABLE_NAME> : Table Name
    """
    config_data, dsl_data = preprocess(**kwargs)
    detect_utils.check_config(config=config_data, required_arguments=['namespace', 'table_name'])
    access_server('post', ctx, 'table/table_info', config_data)


@table.command(short_help="Delete Table Command")
# @click.option('-n', '--namespace', metavar='[NAMESPACE]', help='Namespace')
# @click.option('-t', '--table_name', metavar='[TABLE_NAME]', help='Table Name')
# @click.option('-j', '--job-id', metavar='[JOB_ID]', help='A valid job id')
# @click.option('-r', '--role', metavar='[ROLE]', help='Role')
# @click.option('-p', '--party-id', metavar='[PARTY_ID]', help='Party id')
# @click.option('-cpn', '--component_name', metavar='[COMPONENT_NAME]', help='Component Name')
@cli_args.NAMESPACE
@cli_args.TABLE_NAME
@cli_args.JOBID
@cli_args.ROLE
@cli_args.PARTYID
@cli_args.COMPONENT_NAME
@click.pass_context
def delete(ctx, **kwargs):
    """
    - COMMAND DESCRIPTION:

    Delete A Specified Table.
    """
    # config_data, dsl_data = preprocess(**kwargs)
    # access_server('post', ctx, 'table/delete', config_data)
    click.echo(locals())