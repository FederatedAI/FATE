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

from flow_client.flow_cli.utils import cli_args
from flow_client.flow_cli.utils.cli_utils import (preprocess, download_from_request, access_server, prettify)


@click.group(short_help="Template Operations")
@click.pass_context
def template(ctx):
    """
    \b
    fate template file download
    """
    pass


@template.command("download", short_help="Template Download Command")
@cli_args.MIN_DATA
@cli_args.OUTPUT_PATH
@click.pass_context
def download(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        Download template conf/dsl/data files

    \b
    - USAGE:
        flow template download --min-data 1 --output-path ./examples/
    """
    config_data, dsl_data = preprocess(**kwargs)
    tar_file_name = 'template.tar.gz'
    extract_dir = config_data['output_path']
    with closing(access_server('post', ctx, 'template/download', config_data, False, stream=True)) as response:
        if response.status_code == 200:
            download_from_request(http_response=response, tar_file_name=tar_file_name, extract_dir=extract_dir)
            res = {'retcode': 0,
                   'directory': extract_dir,
                   'retmsg': 'download successfully, please check {} directory'.format(extract_dir)}
        else:
            res = response.json() if isinstance(response, requests.models.Response) else response
    prettify(res)
