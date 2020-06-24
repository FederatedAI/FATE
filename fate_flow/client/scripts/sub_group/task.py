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
import requests
from fate_flow.utils.cli_utils import preprocess, access_server, prettify


@click.group(short_help="Task Operations")
@click.pass_context
def task(ctx):
    """Task Operations Descriptions"""
    pass


@task.command(short_help="List Task")
@click.option('-l', '--limit', default=20, metavar='[LIMIT]', help='Limit count, defaults equals to 20')
@click.pass_context
def list(ctx, **kwargs):
    """
    - COMMAND DESCRIPTION:

    List Task description

    """
    # TODO executed method
    click.echo('Limit number is: %d' % kwargs.get('limit'))


@task.command(short_help="Query Task")
@click.option('-j', '--job_id', metavar="[JOB_ID]", help="Job ID")
@click.option('-p', '--party_id', metavar="[PARTY_ID]", help="Party ID")
@click.option('-r', '--role', metavar="[ROLE]", help="Role")
@click.option('-cpn', '--component_name', metavar="[COMPONENT_NAME]", help="Component Name")
@click.option('-s', '--status', metavar="[STATUS]", help="Job Status")
@click.pass_context
def query(ctx, **kwargs):
    """
    - COMMAND DESCRIPTION:

    Query Task

    """
    config_data, dsl_data = preprocess(**kwargs)
    response = access_server('post', ctx, 'job/task/query', config_data, False)
    prettify(response.json() if isinstance(response, requests.models.Response) else response)
