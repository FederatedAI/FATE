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
import json
import click
import requests
from contextlib import closing
from fate_flow.utils import cli_args
from arch.api.utils import file_utils
from fate_flow.utils.detect_utils import check_config
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
@cli_args.CONF_PATH
@click.pass_context
def load(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        Load Model Command

    \b
    - USAGE:
        flow model load -c fate_flow/examples/publish_load_model.json

    """
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, 'model/load', config_data)


@model.command(short_help="Bind Model Command")
@cli_args.CONF_PATH
@click.pass_context
def bind(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        Bind Model Command

    \b
    - USAGE:
        flow model bind -c fate_flow/examples/bind_model_service.json

    """
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, 'model/bind', config_data)


@model.command("import", short_help="Import Model Command")
@cli_args.CONF_PATH
@click.option("--force", is_flag=True, default=False,
              help="If specified, the existing model file which is named the same as current model's, "
                   "older version of model files would be renamed. Otherwise, importing progress would be rejected.")
@click.option('--from-database', is_flag=True, default=False,
              help="If specified and there is a valid database environment, fate flow will import model from database "
                   "which you specified in configuration file.")
@click.pass_context
def imp(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        Import Model Command. Users can currently import models from files or databases (including mysql and redis).

    \b
    - USAGE:
        flow model import -c fate_flow/examples/import_model.json --force
        flow model import -c fate_flow/examples/restore_model.json --from-database

    """
    config_data, dsl_data = preprocess(**kwargs)
    try:
        required_arguments = ["model_id", "model_version", "initiator", "roles"]
        check_config(config_data, required_arguments)
    except Exception as e:
        click.echo(json.dumps({'retcode': 100, 'retmsg': str(e)}, indent=4))
    else:
        config_data["initiator"] = json.dumps(config_data["initiator"])
        config_data["roles"] = json.dumps(config_data["roles"])

    if not kwargs.pop('from_database'):
        file_path = config_data["file"]
        if not os.path.isabs(file_path):
            file_path = os.path.join(file_utils.get_project_base_directory(), file_path)
        if os.path.exists(file_path):
            files = {'file': open(file_path, 'rb')}
        else:
            raise Exception('The file is obtained from the fate flow client machine, but it does not exist, '
                            'please check the path: {}'.format(file_path))
        access_server('post', ctx, 'model/import', data=config_data, files=files)
    else:
        access_server('post', ctx, 'model/restore', config_data)


@model.command(short_help="Export Model Command")
@cli_args.CONF_PATH
@click.option('--to-database', is_flag=True, default=False,
              help="If specified and there is a valid database environment, fate flow will export model to database "
                   "which you specified in configuration file.")
@click.pass_context
def export(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        Export Model Command. Users can currently export models to files or databases (including mysql and redis).

    \b
    - USAGE:
        flow model export -c fate_flow/examples/export_model.json
        flow model export -c fate_flow/examplse/store_model.json --to-database

    """
    if not kwargs.get('to_database'):
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


# @model.command(short_help="Store Model Command")
# # @click.argument('conf_path', type=click.Path(exists=True), metavar='<CONF_PATH>')
# @cli_args.CONF_PATH
# @click.pass_context
# def store(ctx, **kwargs):
#     """
#     - COMMAND DESCRIPTION:
#
#     Store Model Command
#
#     """
#     config_data, dsl_data = preprocess(**kwargs)
#     access_server('post', ctx, 'model/store', config_data)
#
#
# @model.command(short_help="Restore Model Command")
# # @click.argument('conf_path', type=click.Path(exists=True), metavar='<CONF_PATH>')
# @cli_args.CONF_PATH
# @click.pass_context
# def restore(ctx, **kwargs):
#     """
#     - COMMAND DESCRIPTION:
#
#     Restore Model Command
#
#     """
#     config_data, dsl_data = preprocess(**kwargs)
#     access_server('post', ctx, 'model/restore', config_data)