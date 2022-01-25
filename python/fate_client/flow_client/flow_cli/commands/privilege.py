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


@click.group(short_help="Privilege Operations")
@click.pass_context
def privilege(ctx):
    """
    \b
    Provides numbers of privilege operational commands, including grant, query and delete.
    For more details, please check out the help text.
    """
    pass


@privilege.command("grant", short_help="Grant Privilege Command")
@cli_args.SRC_PARTY_ID
@cli_args.SRC_ROLE
@cli_args.PRIVILEGE_ROLE
@cli_args.PRIVILEGE_COMMAND
@cli_args.PRIVILEGE_COMPONENT
@click.pass_context
def grant(ctx, **kwargs):
    """
    - DESCRIPTION:


    \b
    grant role/command/component privilege

    \b
    - USAGE:
        flow privilege grant --src-party-id 9999 --src-role guest --privilege-role all
    """
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, 'permission/grant/privilege', config_data)


@privilege.command("delete", short_help="Delete Privilege Command")
@cli_args.SRC_PARTY_ID
@cli_args.SRC_ROLE
@cli_args.PRIVILEGE_ROLE
@cli_args.PRIVILEGE_COMMAND
@cli_args.PRIVILEGE_COMPONENT
@click.pass_context
def delete(ctx, **kwargs):
    """
    - DESCRIPTION:


    \b
    delete role/command/component privilege

    \b
    - USAGE:
        flow privilege delete --src-party-id 9999 --src-role guest --privilege-role all
    """
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, 'permission/delete/privilege', config_data)


@privilege.command("query", short_help="Query Privilege Command")
@cli_args.SRC_PARTY_ID
@cli_args.SRC_ROLE
@click.pass_context
def delete(ctx, **kwargs):
    """
    - DESCRIPTION:


    \b
    query role/command/component privilege

    \b
    - USAGE:
        flow privilege query --src-party-id 9999 --src-role guest
    """
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, 'permission/query/privilege', config_data)
