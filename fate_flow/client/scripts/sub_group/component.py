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
import os
import click
import requests
from contextlib import closing
from fate_flow.utils import detect_utils
from fate_flow.utils.cli_utils import preprocess, download_from_request, access_server


@click.group(short_help="Component Operations")
@click.pass_context
def component(ctx):
    """Component Operations"""
    pass


@component.command(short_help="List Components")
@click.option('-l', '--limit', default=20, metavar='<LIMIT>', help='limit count, defaults is 20')
@click.pass_context
def list(ctx, **kwargs):
    """
    - COMMAND DESCRIPTION:

    List Component description

    """
    # TODO executed method
    click.echo('Limit number is: %d' % kwargs.get('limit'))


@component.command(short_help="Component Metrics")
@click.argument('job_id', metavar="<JOB_ID>")
@click.argument('role', metavar="<ROLE>")
@click.argument('party_id', metavar="<PARTY_ID>")
@click.argument('component_name', metavar="<COMPONENT_NAME>")
@click.pass_context
def metrics(ctx, **kwargs):
    """
    - COMMAND DESCRIPTION:


    - REQUIRED ARGUMENTS:

    \b
    <JOB_ID> : Job ID
    <ROLE> : Role
    <PARTY_ID> : Party ID
    <COMPONENT_NAME> : Component Name
    """
    config_data, dsl_data = preprocess(**kwargs)
    detect_utils.check_config(config=config_data,
                              required_arguments=['job_id', 'component_name', 'role', 'party_id'])
    access_server('post', ctx, 'tracking/component/metrics', config_data)


@component.command(short_help="Component Metric All")
@click.argument('job_id', metavar="<JOB_ID>")
@click.argument('role', metavar="<ROLE>")
@click.argument('party_id', metavar="<PARTY_ID>")
@click.argument('component_name', metavar="<COMPONENT_NAME>")
@click.pass_context
def metric_all(ctx, **kwargs):
    """
    - COMMAND DESCRIPTION:

    Query all metric data.

    - REQUIRED ARGUMENTS:

    \b
    <JOB_ID> : Job ID
    <ROLE> : Role
    <PARTY_ID> : Party ID
    <COMPONENT_NAME> : Component Name
    """
    config_data, dsl_data = preprocess(**kwargs)
    detect_utils.check_config(config=config_data,
                              required_arguments=['job_id', 'component_name', 'role', 'party_id'])
    access_server('post', ctx, 'tracking/component/metric/all', config_data)


@component.command(short_help="Component Parameters")
@click.argument('job_id', metavar="<JOB_ID>")
@click.argument('role', metavar="<ROLE>")
@click.argument('party_id', metavar="<PARTY_ID>")
@click.argument('component_name', metavar="<COMPONENT_NAME>")
@click.pass_context
def parameters(ctx, **kwargs):
    """
    - COMMAND DESCRIPTION:

    Query the parameters of this component.

    - REQUIRED ARGUMENTS:

    \b
    <JOB_ID> : Job ID
    <ROLE> : Role
    <PARTY_ID> : Party ID
    <COMPONENT_NAME> : Component Name
    """
    config_data, dsl_data = preprocess(**kwargs)
    detect_utils.check_config(config=config_data,
                              required_arguments=['job_id', 'component_name', 'role', 'party_id'])
    access_server('post', ctx, 'tracking/component/parameters', config_data)


@component.command(short_help="Component Output Data")
@click.argument('job_id', metavar="<JOB_ID>")
@click.argument('role', metavar="<ROLE>")
@click.argument('party_id', metavar="<PARTY_ID>")
@click.argument('component_name', metavar="<COMPONENT_NAME>")
@click.argument('output_path', type=click.Path(), metavar="<OUTPUT_PATH>")
@click.option('-l', '--limit', metavar="[LIMIT]", help='limit count, defaults is 20')
@click.pass_context
def output_data(ctx, **kwargs):
    """
    - COMMAND DESCRIPTION:

    Download the output data of this component.

    - REQUIRED ARGUMENTS:

    \b
    <JOB_ID> : Job ID
    <ROLE> : Role
    <PARTY_ID> : Party ID
    <COMPONENT_NAME> : Component Name
    <OUTPUT_PATH> : Config Output Path
    """
    config_data, dsl_data = preprocess(**kwargs)
    detect_utils.check_config(config=config_data,
                              required_arguments=['job_id', 'component_name', 'role', 'party_id', 'output_path'])
    tar_file_name = 'job_{}_{}_{}_{}_output_data.tar.gz'.format(config_data['job_id'],
                                                                config_data['component_name'],
                                                                config_data['role'],
                                                                config_data['party_id'])
    extract_dir = os.path.join(config_data['output_path'], tar_file_name.replace('.tar.gz', ''))
    with closing(access_server('get', ctx, 'tracking/component/output/data/download',
                               config_data, False, stream=True)) as response:
        if response.status_code == 200:
            try:
                download_from_request(http_response=response, tar_file_name=tar_file_name, extract_dir=extract_dir)
                response = {'retcode': 0,
                            'directory': extract_dir,
                            'retmsg': 'download successfully, please check {} directory'.format(extract_dir)}
            except:
                response = {'retcode': 100,
                            'retmsg': 'download failed, please check if the parameters are correct'}
        else:
            response = response.json()
    click.echo(response.json() if isinstance(response, requests.models.Response) else response)


@component.command(short_help="Component Output Model")
@click.argument('job_id', metavar="<JOB_ID>")
@click.argument('role', metavar="<ROLE>")
@click.argument('party_id', metavar="<PARTY_ID>")
@click.argument('component_name', metavar="<COMPONENT_NAME>")
@click.pass_context
def output_model(ctx, **kwargs):
    """
    - COMMAND DESCRIPTION:

    Query this component model.

    - REQUIRED ARGUMENTS:

    \b
    <JOB_ID> : Job ID
    <ROLE> : Role
    <PARTY_ID> : Party ID
    <COMPONENT_NAME> : Component Name
    """
    config_data, dsl_data = preprocess(**kwargs)
    detect_utils.check_config(config=config_data,
                              required_arguments=['job_id', 'component_name', 'role', 'party_id'])
    access_server('post', ctx, 'tracking/component/output/model', config_data)


@component.command(short_help="Component Output Data Table")
@click.argument('job_id', metavar="<JOB_ID>")
@click.argument('role', metavar="<ROLE>")
@click.argument('party_id', metavar="<PARTY_ID>")
@click.argument('component_name', metavar="<COMPONENT_NAME>")
@click.pass_context
def output_data_table(ctx, **kwargs):
    """
    - COMMAND DESCRIPTION:

    View table name and namespace.

    - REQUIRED ARGUMENTS:

    \b
    <JOB_ID> : Job ID
    <ROLE> : Role
    <PARTY_ID> : Party ID
    <COMPONENT_NAME> : Component Name
    """
    config_data, dsl_data = preprocess(**kwargs)
    detect_utils.check_config(config=config_data,
                              required_arguments=['job_id', 'component_name', 'role', 'party_id'])
    access_server('post', ctx, 'tracking/component/output/data/table', config_data)
