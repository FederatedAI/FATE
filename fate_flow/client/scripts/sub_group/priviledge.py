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
from fate_flow.utils import detect_utils
from fate_flow.utils.cli_utils import preprocess, access_server


@click.group(short_help="Priviledge Operations")
@click.pass_context
def priviledge(ctx):
    """Priviledge Operations Descriptions"""
    pass


@priviledge.command(short_help="Query priviledge")
@click.argument('src_party_id', metavar='<SRC_PARTY_ID>')
@click.argument('src_role', metavar='<SRC_ROLE>')
@click.pass_context
def query(ctx, **kwargs):
    """
    - COMMAND DESCRIPTION:

    Query priviledge information.

    - REQUIRED ARGUMENTS:

    \b
    <SRC_PARTY_ID> : Source Party ID
    <SRC_ROLE> : Source Role
    """
    config_data, dsl_data = preprocess(**kwargs)
    detect_utils.check_config(config=config_data, required_arguments=['src_party_id', 'src_role'])
    access_server('post', ctx, 'permission/query/privilege', config_data)


@priviledge.command(short_help="Grant priviledge")
@click.argument('src_party_id', metavar='<SRC_PARTY_ID>')
@click.argument('src_role', metavar='<SRC_ROLE>')
@click.pass_context
def grant(ctx, **kwargs):
    """
    - COMMAND DESCRIPTION:

    Grant priviledge.

    - REQUIRED ARGUMENTS:

    \b
    <SRC_PARTY_ID> : Source Party ID
    <SRC_ROLE> : Source Role
    """
    config_data, dsl_data = preprocess(**kwargs)
    detect_utils.check_config(config=config_data, required_arguments=['src_party_id', 'src_role'])
    access_server('post', ctx, 'permission/grant/privilege', config_data)


@priviledge.command(short_help="Delete priviledge")
@click.argument('src_party_id', metavar='<SRC_PARTY_ID>')
@click.argument('src_role', metavar='<SRC_ROLE>')
@click.pass_context
def delete(ctx, **kwargs):
    """
    - COMMAND DESCRIPTION:

    Delete priviledge.

    - REQUIRED ARGUMENTS:

    \b
    <SRC_PARTY_ID> : Source Party ID
    <SRC_ROLE> : Source Role
    """
    config_data, dsl_data = preprocess(**kwargs)
    detect_utils.check_config(config=config_data, required_arguments=['src_party_id', 'src_role'])
    access_server('post', ctx, 'permission/delete/privilege', config_data)
