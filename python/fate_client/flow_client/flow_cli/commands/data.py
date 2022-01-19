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
from flow_client.flow_cli.utils import cli_args
from flow_client.flow_cli.utils.cli_utils import preprocess, access_server, check_abs_path, prettify
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
import json


@click.group(short_help="Data Operations")
@click.pass_context
def data(ctx):
    """
    \b
    Provides numbers of data operational commands, including upload, download and etc.
    For more details, please check out the help text.
    """
    pass


@data.command("upload", short_help="Upload Table Command")
@cli_args.CONF_PATH
@click.option('--verbose', is_flag=True, default=False,
              help="If specified, verbose mode will be turn on. "
                   "Users can have feedback on upload task in progress. (default: False)")
@click.option('--drop', is_flag=True, default=False,
              help="If specified, data of old version would be replaced by the current version. "
                   "Otherwise, current upload task would be rejected. (default: False)")
@click.pass_context
def upload(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        Upload Data Table.

    \b
    - Usage:
        flow data upload -c fateflow/examples/upload/upload_guest.json
        flow data upload -c fateflow/examples/upload/upload_host.json --verbose --drop
    """
    kwargs['drop'] = 1 if kwargs['drop'] else 0
    kwargs['verbose'] = int(kwargs['verbose'])
    config_data, dsl_data = preprocess(**kwargs)
    if config_data.get('use_local_data', 1):
        file_name = check_abs_path(config_data.get('file'))
        if os.path.exists(file_name):
            with open(file_name, 'rb') as fp:
                data = MultipartEncoder(
                    fields={'file': (os.path.basename(file_name), fp, 'application/octet-stream')}
                )
                tag = [0]

                def read_callback(monitor):
                    if config_data.get('verbose') == 1:
                        sys.stdout.write("\r UPLOADING:{0}{1}".format(
                            "|" * (monitor.bytes_read * 100 // monitor.len), '%.2f%%' % (monitor.bytes_read * 100 // monitor.len)))
                        sys.stdout.flush()
                        if monitor.bytes_read / monitor.len == 1:
                            tag[0] += 1
                            if tag[0] == 2:
                                sys.stdout.write('\n')

                data = MultipartEncoderMonitor(data, read_callback)
                access_server('post', ctx, 'data/upload', json_data=None, data=data,
                              params=json.dumps(config_data), headers={'Content-Type': data.content_type})
        else:
            prettify(
                {
                    "retcode": 100,
                    "retmsg": "The file is obtained from the fate flow client machine, but it does not exist, "
                              "please check the path: {}".format(file_name)
                }
            )
    else:
        access_server('post', ctx, 'data/upload', config_data)


@data.command("download", short_help="Download Table Command")
@cli_args.CONF_PATH
@click.pass_context
def download(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        Download Data Table.

    \b
    - Usage:
        flow data download -c fateflow/examples/download/download_table.json
    """
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, "data/download", config_data)


@data.command("writer", short_help="write Table Command")
@cli_args.CONF_PATH
@click.pass_context
def writer(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        Download Data Table.

    \b
    - Usage:
        flow data download -c fateflow/examples/writer/external_storage.json
    """
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, "data/writer", config_data)


@data.command("upload-history", short_help="Upload History Command")
@cli_args.LIMIT
@cli_args.JOBID
@click.pass_context
def upload_history(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        Query Upload Table History.

    \b
    - USAGE:
        flow data upload-history -l 20
        flow data upload-history --job-id $JOB_ID
    """
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, "data/upload/history", config_data)


# @data.command(short_help="")
@click.pass_context
def download_history(ctx):
    """

    """
    pass
