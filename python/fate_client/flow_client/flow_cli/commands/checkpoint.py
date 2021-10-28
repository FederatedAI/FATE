#
#  Copyright 2021 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the 'License');
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an 'AS IS' BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import sys

import click

from flow_client.flow_cli.utils import cli_args
from flow_client.flow_cli.utils.cli_utils import preprocess, access_server


@click.group(short_help='Checkpoint Operations')
@click.pass_context
def checkpoint(ctx, **kwargs):
    pass


@checkpoint.command('list', short_help='List checkpoints')
@cli_args.ROLE_REQUIRED
@cli_args.PARTYID_REQUIRED
@cli_args.MODEL_ID_REQUIRED
@cli_args.MODEL_VERSION_REQUIRED
@cli_args.COMPONENT_NAME_REQUIRED
@click.pass_context
def list_checkpoints(ctx, **kwargs):
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, 'checkpoint/list', config_data)


@checkpoint.command('get', short_help='Get a checkpoint by step_index or step_name')
@cli_args.ROLE_REQUIRED
@cli_args.PARTYID_REQUIRED
@cli_args.MODEL_ID_REQUIRED
@cli_args.MODEL_VERSION_REQUIRED
@cli_args.COMPONENT_NAME_REQUIRED
@click.option('--step-index', help='Step index', type=click.INT)
@click.option('--step-name', help='Step name', type=click.STRING)
@click.pass_context
def get_checkpoint(ctx, **kwargs):
    config_data, dsl_data = preprocess(**kwargs)
    if len(config_data.keys() & {'step_index', 'step_name'}) != 1:
        click.echo("Error: Missing option '--step-index' or '--step-name'.", err=True)
        sys.exit(2)
    access_server('post', ctx, 'checkpoint/get', config_data)
