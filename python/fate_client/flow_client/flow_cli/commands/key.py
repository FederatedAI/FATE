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


@click.group(short_help="Key Operations")
@click.pass_context
def key(ctx):
    """
    \b
    Provides numbers of key operational commands, including save, query and delete.
    For more details, please check out the help text.
    """
    pass


@key.command("save", short_help="Save Public Key Command")
@cli_args.CONF_PATH
@click.pass_context
def save(ctx, **kwargs):
    """
    - DESCRIPTION:


    \b
    save other site public key

    \b
    - USAGE:
        flow key save -c fateflow/examples/key/save_public_key.json
    """
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, 'key/public/save', config_data)


@key.command("delete", short_help="Delete Public Key Command")
@cli_args.PARTYID_REQUIRED
@click.pass_context
def delete(ctx, **kwargs):
    """
    - DESCRIPTION:


    \b
    delete other site public key

    \b
    - USAGE:
        flow key delete  -p 10000
    """
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, 'key/public/delete', config_data)


@key.command("query", short_help="Query Public Key Command")
@cli_args.PARTYID_REQUIRED
@click.pass_context
def query(ctx, **kwargs):
    """
    - DESCRIPTION:


    \b
     query site public key

    \b
    - USAGE:
        flow key query -p 10000
    """
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, 'key/query', config_data)
