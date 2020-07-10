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
import os
import sys
import traceback
from arch.api.utils import file_utils
from fate_flow.utils.cli_utils import (preprocess, access_server, prettify,
                                       start_cluster_standalone_job_server)
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor


@click.group(short_help="Data Operations")
@click.pass_context
def data(ctx):
    """
    \b
    Provides numbers of data operational commands, including upload, download and etc.
    For more details, please check out the help text.
    """
    pass


@data.command(short_help="Upload Table Command")
@click.argument('conf_path', type=click.Path(exists=True), metavar='<CONF_PATH>')
@click.option('-v', '--verbose', type=click.Choice(['0', '1']), default='0', metavar="[VERBOSE]",
              help="Verbose mode, 0 means 'disable'(default), 1 means 'enable'")
@click.option('-d', '--drop', type=click.Choice(['0', '1']), metavar="[DROP]", default=0,
              help="Drop data before uploading. If 'drop' is set to be 0 (defualt), when data had been uploaded before,"
                   " current upload task would be rejected. If 'drop' is set to be 1, data of old version would be "
                   "replaced by the latest version.")
@click.pass_context
def upload(ctx, **kwargs):
    """
    - COMMAND DESCRIPTION:

    Upload Data Table.

    - REQUIRED ARGUMENTS:

    \b
    <CONF_PATH> : Configuration file path

    """
    kwargs['drop'] = int(kwargs['drop']) if int(kwargs['drop']) else 2
    kwargs['verbose'] = int(kwargs['verbose'])
    config_data, dsl_data = preprocess(**kwargs)
    if config_data.get('use_local_data', 1):
        file_name = config_data.get('file')
        if not os.path.isabs(file_name):
            file_name = os.path.join(file_utils.get_project_base_directory(), file_name)
        if os.path.exists(file_name):
            with open(file_name, 'rb') as fp:
                data = MultipartEncoder(
                    fields={'file': (os.path.basename(file_name), fp, 'application/octet-stream')}
                )
                tag = [0]

                def read_callback(monitor):
                    if config_data.get('verbose') == 1:
                        sys.stdout.write("\r UPLOADING:{0}{1}".format("|" * (monitor.bytes_read * 100 // monitor.len), '%.2f%%' % (monitor.bytes_read * 100 // monitor.len)))
                        sys.stdout.flush()
                        if monitor.bytes_read / monitor.len == 1:
                            tag[0] += 1
                            if tag[0] == 2:
                                sys.stdout.write('\n')

                data = MultipartEncoderMonitor(data, read_callback)
                response = access_server('post', ctx, 'data/upload', json=None, echo=False, data=data,
                                         params=config_data,
                                         headers={'Content-Type': data.content_type})
        else:
            raise Exception('The file is obtained from the fate flow client machine, but it does not exist, '
                            'please check the path: {}'.format(file_name))
    else:
        response = access_server('post', ctx, 'data/upload', config_data, False)
    try:
        if response.json()['retcode'] == 999:
            start_cluster_standalone_job_server()
            access_server('post', ctx, "data/upload", config_data)
        else:
            prettify(response.json())
    except:
        click.echo(traceback.format_exc())


@data.command(short_help="Download Table Command")
@click.argument('conf_path', type=click.Path(exists=True), metavar='<CONF_PATH>')
@click.pass_context
def download(ctx, **kwargs):
    """
    - COMMAND DESCRIPTION:

    Download Data Table.

    - REQUIRED ARGUMENTS:

    \b
    <CONF_PATH> : Configuration file path
    """
    config_data, dsl_data = preprocess(**kwargs)
    response = access_server('post', ctx, "data/download", config_data, False)
    try:
        if response.json()['retcode'] == 999:
            start_cluster_standalone_job_server()
            access_server('post', ctx, "data/download", config_data)
        else:
            prettify(response.json())
    except:
        click.echo(traceback.format_exc())


@data.command(short_help="Upload History Command")
@click.option('-l', '--limit', metavar="[LIMIT]", default=10, help="Limit count, defaults is 10")
@click.option('-j', '--job_id', metavar="[JOB_ID]", help="Job ID")
@click.pass_context
def upload_history(ctx, **kwargs):
    """
    - COMMAND DESCRIPTION:

    \b
    Query Upload Table History.
    """
    config_data, dsl_data = preprocess(**kwargs)
    response = access_server('post', ctx, "data/upload/history", config_data, False)

    try:
        if response.json()['retcode'] == 999:
            start_cluster_standalone_job_server()
            access_server('post', ctx, "data/upload/history", config_data)
        else:
            prettify(response.json())
    except:
        click.echo(traceback.format_exc())


# @data.command(short_help="")
@click.pass_context
def download_history():
    """

    """
    pass


