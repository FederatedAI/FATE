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
import json
import time
import click
import requests
from contextlib import closing

from fate_flow.utils import detect_utils
from fate_flow.utils.cli_utils import (preprocess, download_from_request,
                                       access_server, prettify)


@click.group(short_help="Job Operations")
@click.pass_context
def job(ctx):
    """
    Provides a bunch of Job Operations Methods
    """
    pass


@job.command(short_help="Submit job description")
@click.argument('workmode', type=click.Choice(['0', '1']), metavar='<WORKMODE>')
@click.argument('conf_path', type=click.Path(exists=True), metavar='<CONF_PATH>')
@click.argument('dsl_path', type=click.Path(exists=True), metavar='<DSL_PATH>')
@click.pass_context
def submit(ctx, **kwargs):
    """
    - METHOD DESCRIPTION:

    \b
    Submit a pipeline job.
    Used to be 'submit_job'.

    - REQUIRED ARGUMENTS:

    \b
    <WORKMODE> : Workmode 0(standalone) or 1(cluster).
    <CONF_PATH> : Configuration file path.
    <DSL_PATH> : Dsl file path.
    """
    config_data, dsl_data = preprocess(**kwargs)
    post_data = {
        'job_dsl': dsl_data,
        'job_runtime_conf': config_data
    }

    response = access_server('post', ctx, 'job/submit', post_data, False)

    try:
        if response.json()['retcode'] == 999:
            click.echo('use service.sh to start standalone node server....')
            os.system('sh service.sh start --standalone_node')
            time.sleep(5)
            access_server('post', ctx, 'job/submit', post_data)
    except:
        pass


@job.command(short_help="List job")
@click.option('-l', '--limit', default=20, metavar='[LIMIT]', help='Limit count, defaults is 20')
@click.pass_context
def list(ctx, **kwargs):
    """
    - COMMAND DESCRIPTION:

    List job description

    """
    # TODO executed method
    click.echo('Limit number is: %d' % kwargs.get('limit'))


@job.command(short_help="Query job")
@click.option('-j', '--job_id', metavar="[JOB_ID]", help="Job ID")
@click.option('-r', '--role', metavar="[ROLE]", help="Role")
@click.option('-p', '--party_id', metavar="[PARTY_ID]", help="Party ID")
@click.option('-cpn', '--component_name', metavar="[COMPONENT_NAME]", help="Component Name")
@click.option('-s', '--status', metavar="[STATUS]", help="Job Status")
@click.pass_context
def query(ctx, **kwargs):
    """
    - COMMAND DESCRIPTION:

    \b
    Query job information by filters.
    Used to be 'query_job'
    """
    config_data, dsl_data = preprocess(**kwargs)
    response = access_server('post', ctx, "job/query", config_data, False)
    if isinstance(response, requests.models.Response):
        response = response.json()
    if response['retcode'] == 0:
        for i in range(len(response['data'])):
            del response['data'][i]['f_runtime_conf']
            del response['data'][i]['f_dsl']
    prettify(response.json() if isinstance(response, requests.models.Response) else response)


@job.command(short_help="Clean job")
@click.option('-j', '--job_id', metavar="[JOB_ID]", help="Job ID")
@click.option('-r', '--role', metavar="[ROLE]", help="Role")
@click.option('-p', '--party_id', metavar="[PARTY_ID]", help="Party ID")
@click.option('-cpn', '--component_name', metavar="[COMPONENT_NAME]", help="Component Name")
@click.pass_context
def clean(ctx, **kwargs):
    """
    - COMMAND DESCRIPTION:

    \b
    Clean job description
    Used to be 'clean_job'
    """
    config_data, dsl_data = preprocess(**kwargs)
    detect_utils.check_config(config=config_data, required_arguments=['job_id'])
    access_server('post', ctx, "job/clean", config_data)


@job.command(short_help="Stop job")
@click.argument('job_id', metavar="<JOB_ID>")
@click.pass_context
def stop(ctx, **kwargs):
    """
    - COMMAND DESCRIPTION:

    Stop job description

    - REQUIRED ARGUMENTS:

    \b
    <JOB_ID> : Provides a valid job id.
    """
    config_data, dsl_data = preprocess(**kwargs)
    detect_utils.check_config(config=config_data, required_arguments=['job_id'])
    access_server('post', ctx, "job/stop", config_data)


@job.command(short_help="Config job")
@click.argument('job_id', metavar="<JOB_ID>")
@click.argument('role', metavar="<ROLE>")
@click.argument('party_id', metavar="<PARTY_ID>")
@click.argument('output_path', metavar="<OUTPUT_PATH>")
@click.pass_context
def config(ctx, **kwargs):
    """
    - COMMAND DESCRIPTION:

    Config job description

    - REQUIRED ARGUMENTS:

    \b
    <JOB_ID> : Job ID
    <ROLE> : Role
    <PARTY_ID> : Party ID
    <OUTPUT_PATH> : Output path
    """
    config_data, dsl_data = preprocess(**kwargs)
    detect_utils.check_config(config=config_data, required_arguments=['job_id', 'role', 'party_id', 'output_path'])
    response = access_server('post', ctx, 'job/config', config_data, False).json()
    if response['retcode'] == 0:
        job_id = response['data']['job_id']
        download_directory = os.path.join(config_data['output_path'], 'job_{}_config'.format(job_id))
        os.makedirs(download_directory, exist_ok=True)
        for k, v in response['data'].items():
            if k == 'job_id':
                continue
            with open('{}/{}.json'.format(download_directory, k), 'w') as fw:
                json.dump(v, fw, indent=4)
        del response['data']['dsl']
        del response['data']['runtime_conf']
        response['directory'] = download_directory
        response['retmsg'] = 'download successfully, please check {} directory'.format(download_directory)
    prettify(response.json() if isinstance(response, requests.models.Response) else response)


@job.command(short_help="Log job")
@click.argument('job_id', metavar="<JOB_ID>")
@click.argument('output_path', metavar="<OUTPUT_PATH>")
@click.pass_context
def log(ctx, **kwargs):
    """
    - COMMAND DESCRIPTION:

    Log job description

    - REQUIRED ARGUMENTS:

    \b
    <JOB_ID> : Job ID
    <OUTPUT_PATH> : Output path
    """
    config_data, dsl_data = preprocess(**kwargs)
    detect_utils.check_config(config=config_data, required_arguments=['job_id', 'output_path'])
    job_id = config_data['job_id']
    tar_file_name = 'job_{}_log.tar.gz'.format(job_id)
    extract_dir = os.path.join(config_data['output_path'], 'job_{}_log'.format(job_id))
    with closing(access_server('get', ctx, 'job/log', config_data, False, stream=True)) as response:
        if response.status_code == 200:
            download_from_request(http_response=response, tar_file_name=tar_file_name, extract_dir=extract_dir)
            response = {'retcode': 0,
                        'directory': extract_dir,
                        'retmsg': 'download successfully, please check {} directory'.format(extract_dir)}
        else:
            response = response.json()
    prettify(response.json() if isinstance(response, requests.models.Response) else response)
