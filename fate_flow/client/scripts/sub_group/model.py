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
import re
import click
import requests
from contextlib import closing
from arch.api.utils import file_utils
from fate_flow.utils.cli_utils import preprocess, access_server


@click.group(short_help="Model Operations")
@click.pass_context
def model(ctx):
    """
    \b
    Provides numbers of model operational commands, including load, store, import and etc.
    For more details, please check out the help text.
    """
    pass


@model.command(short_help="Load Model Command")
@click.argument('conf_path', type=click.Path(exists=True), metavar='<CONF_PATH>')
@click.pass_context
def load(ctx, **kwargs):
    """
    - COMMAND DESCRIPTION:

    Load Model Command

    - REQUIRED ARGUMENTS:

    \b
    <CONF_PATH> : Configuration File Path
    """
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, 'model/load', config_data)


@model.command(short_help="Bind Model Command")
@click.argument('conf_path', type=click.Path(exists=True), metavar='<CONF_PATH>')
@click.pass_context
def bind(ctx, **kwargs):
    """
    - COMMAND DESCRIPTION:

    Bind Model Command

    - REQUIRED ARGUMENTS:

    \b
    <CONF_PATH> : Configuration File Path
    """
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, 'model/bind', config_data)


@model.command(short_help="Store Model Command")
@click.argument('conf_path', type=click.Path(exists=True), metavar='<CONF_PATH>')
@click.pass_context
def store(ctx, **kwargs):
    """
    - COMMAND DESCRIPTION:

    Store Model Command

    - REQUIRED ARGUMENTS:

    \b
    <CONF_PATH> : Configuration File Path
    """
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, 'model/store', config_data)


@model.command(short_help="Restore Model Command")
@click.argument('conf_path', type=click.Path(exists=True), metavar='<CONF_PATH>')
@click.pass_context
def restore(ctx, **kwargs):
    """
    - COMMAND DESCRIPTION:

    Restore Model Command

    - REQUIRED ARGUMENTS:

    \b
    <CONF_PATH> : Configuration File Path
    """
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, 'model/restore', config_data)


# TODO Rename this method
@model.command(short_help="Import Model Command")
@click.argument('conf_path', type=click.Path(exists=True), metavar='<CONF_PATH>')
@click.argument('type', type=click.Choice(['0', '1']), default='0', metavar='<TYPE>')
@click.pass_context
def imp(ctx, **kwargs):
    """
    - COMMAND DESCRIPTION:

    Import Model Command

    - REQUIRED ARGUMENTS:

    \b
    <CONF_PATH> : Configuration File Path
    <TYPE> : Choose import type from 0(default) and 1.
    0 means import from files, while 1 means import from database which you can specify in configuration file.
    """
    if not kwargs.get('type'):
        config_data, dsl_data = preprocess(**kwargs)
        file_path = config_data["file"]
        if not os.path.isabs(file_path):
            file_path = os.path.join(file_utils.get_project_base_directory(), file_path)
        if os.path.exists(file_path):
            files = {'file': open(file_path, 'rb')}
        else:
            raise Exception('The file is obtained from the fate flow client machine, but it does not exist, '
                            'please check the path: {}'.format(file_path))
        access_server('post', ctx, 'model/import', config_data, files=files)
    else:
        config_data, dsl_data = preprocess(**kwargs)
        access_server('post', ctx, 'model/restore', config_data)


@model.command(short_help="Export Model Command")
@click.argument('conf_path', type=click.Path(exists=True), metavar='<CONF_PATH>')
@click.argument('type', type=click.Choice(['0', '1']), default='0', metavar='<TYPE>')
@click.pass_context
def export(ctx, **kwargs):
    """
    - COMMAND DESCRIPTION:

    Export Model Command

    - REQUIRED ARGUMENTS:

    \b
    <CONF_PATH> : Configuration File Path
    <TYPE> : Choose export type from 0(default) and 1.
    0 means export to files, while 1 means export to database which you can specify in configuration file.
    """
    if not kwargs.get('type'):
        config_data, dsl_data = preprocess(**kwargs)
        with closing(access_server('get', ctx, 'model/export', config_data, False, stream=True)) as response:
            if response.status_code == 200:
                archive_file_name = re.findall("filename=(.+)", response.headers["Content-Disposition"])[0]
                os.makedirs(config_data["output_path"], exist_ok=True)
                archive_file_path = os.path.join(config_data["output_path"], archive_file_name)
                with open(archive_file_path, 'wb') as fw:
                    for chunk in response.iter_content(1024):
                        if chunk:
                            fw.write(chunk)
                response = {'retcode': 0,
                            'file': archive_file_path,
                            'retmsg': 'download successfully, please check {}'.format(archive_file_path)}
            else:
                response = response.json()
        click.echo(response.json() if isinstance(response, requests.models.Response) else response)
    else:
        config_data, dsl_data = preprocess(**kwargs)
        access_server('post', ctx, 'model/store', config_data)
