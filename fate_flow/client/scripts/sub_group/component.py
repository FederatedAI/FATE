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
from contextlib import closing
from fate_flow.utils import detect_utils
from fate_flow.utils.cli_utils import prettify, preprocess, download_from_request, access_server


@click.group(short_help="Component Operations")
@click.pass_context
def component(ctx):
    """
    \b
    Provides numbers of component operational commands, including metrics, parameters and etc.
    For more details, please check out the help text.
    """
    pass


@component.command(short_help="List Components Command")
@click.argument('job_id', metavar="<JOB_ID>")
@click.pass_context
def list(ctx, **kwargs):
    """
    - COMMAND DESCRIPTION:

    List components of a specified job.

    - REQUIRED ARGUMENTS:

    \b
    <JOB_ID> : A valid job id

    """
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, 'tracking/component/list', config_data)


@component.command(short_help="Component Metrics Command")
@click.argument('job_id', metavar="<JOB_ID>")
@click.argument('role', metavar="<ROLE>")
@click.argument('party_id', metavar="<PARTY_ID>")
@click.argument('component_name', metavar="<COMPONENT_NAME>")
@click.pass_context
def metrics(ctx, **kwargs):
    """
    - COMMAND DESCRIPTION:

    Query the List of Metrics.

    - REQUIRED ARGUMENTS:

    \b
    <JOB_ID> : A valid job id
    <ROLE> : Role
    <PARTY_ID> : Party ID
    <COMPONENT_NAME> : Component Name
    """
    config_data, dsl_data = preprocess(**kwargs)
    detect_utils.check_config(config=config_data,
                              required_arguments=['job_id', 'component_name', 'role', 'party_id'])
    access_server('post', ctx, 'tracking/component/metrics', config_data)


@component.command(short_help="Component Metric All Command")
@click.argument('job_id', metavar="<JOB_ID>")
@click.argument('role', metavar="<ROLE>")
@click.argument('party_id', metavar="<PARTY_ID>")
@click.argument('component_name', metavar="<COMPONENT_NAME>")
@click.pass_context
def metric_all(ctx, **kwargs):
    """
    - COMMAND DESCRIPTION:

    Query All Metric Data.

    - REQUIRED ARGUMENTS:

    \b
    <JOB_ID> : A valid job id
    <ROLE> : Role
    <PARTY_ID> : Party ID
    <COMPONENT_NAME> : Component Name
    """
    config_data, dsl_data = preprocess(**kwargs)
    detect_utils.check_config(config=config_data,
                              required_arguments=['job_id', 'component_name', 'role', 'party_id'])
    access_server('post', ctx, 'tracking/component/metric/all', config_data)


@component.command(short_help="Delete Metric Command")
@click.option('-d', '--date', metavar="[DATE]", help="An 8-Digit Valid Date, Format Like 'YYYYMMDD'")
@click.option('-j', '--job_id', metavar="[JOB_ID]", help="Job ID")
@click.pass_context
def metric_delete(ctx, **kwargs):
    """
    - COMMAND DESCRIPTION:

    \b
    Delete specified metric.
    If you input two optional argument, the 'date' argument will be detected in priority.
    """
    config_data, dsl_data = preprocess(**kwargs)
    if config_data.get('date'):
        config_data['model'] = config_data.pop('date')
    access_server('post', ctx, 'tracking/component/metric/delete', config_data)


@component.command(short_help="Component Parameters Command")
@click.argument('job_id', metavar="<JOB_ID>")
@click.argument('role', metavar="<ROLE>")
@click.argument('party_id', metavar="<PARTY_ID>")
@click.argument('component_name', metavar="<COMPONENT_NAME>")
@click.pass_context
def parameters(ctx, **kwargs):
    """
    - COMMAND DESCRIPTION:

    Query the parameters of a specified component.

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


@component.command(short_help="Component Output Data Command")
@click.argument('job_id', metavar="<JOB_ID>")
@click.argument('role', metavar="<ROLE>")
@click.argument('party_id', metavar="<PARTY_ID>")
@click.argument('component_name', metavar="<COMPONENT_NAME>")
@click.argument('output_path', type=click.Path(), metavar="<OUTPUT_PATH>")
@click.option('-l', '--limit', metavar="[LIMIT]", default=10, help='limit count, defaults is 10')
@click.pass_context
def output_data(ctx, **kwargs):
    """
    - COMMAND DESCRIPTION:

    Download the Output Data of A Specified Component.

    - REQUIRED ARGUMENTS:

    \b
    <JOB_ID> : A valid job id
    <ROLE> : Role
    <PARTY_ID> : Party ID
    <COMPONENT_NAME> : Component Name
    <OUTPUT_PATH> : Output directory path
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
    prettify(response)


@component.command(short_help="Component Output Model Command")
@click.argument('job_id', metavar="<JOB_ID>")
@click.argument('role', metavar="<ROLE>")
@click.argument('party_id', metavar="<PARTY_ID>")
@click.argument('component_name', metavar="<COMPONENT_NAME>")
@click.pass_context
def output_model(ctx, **kwargs):
    """
    - COMMAND DESCRIPTION:

    Query the Model of A Speicied Component.

    - REQUIRED ARGUMENTS:

    \b
    <JOB_ID> : A valid job id
    <ROLE> : Role
    <PARTY_ID> : Party ID
    <COMPONENT_NAME> : Component Name
    """
    config_data, dsl_data = preprocess(**kwargs)
    detect_utils.check_config(config=config_data,
                              required_arguments=['job_id', 'component_name', 'role', 'party_id'])
    access_server('post', ctx, 'tracking/component/output/model', config_data)


@component.command(short_help="Component Output Data Table Command")
@click.argument('job_id', metavar="<JOB_ID>")
@click.argument('role', metavar="<ROLE>")
@click.argument('party_id', metavar="<PARTY_ID>")
@click.argument('component_name', metavar="<COMPONENT_NAME>")
@click.pass_context
def output_data_table(ctx, **kwargs):
    """
    - COMMAND DESCRIPTION:

    View Table Name and Namespace.

    - REQUIRED ARGUMENTS:

    \b
    <JOB_ID> : A valid job id
    <ROLE> : Role
    <PARTY_ID> : Party ID
    <COMPONENT_NAME> : Component Name
    """
    config_data, dsl_data = preprocess(**kwargs)
    detect_utils.check_config(config=config_data,
                              required_arguments=['job_id', 'component_name', 'role', 'party_id'])
    access_server('post', ctx, 'tracking/component/output/data/table', config_data)
