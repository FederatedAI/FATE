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
from flow_client.flow_cli.utils import cli_args
from contextlib import closing
from flow_client.flow_cli.utils.cli_utils import preprocess, access_server, prettify, get_project_base_directory


@click.group(short_help="Model Operations")
@click.pass_context
def model(ctx):
    """
    \b
    Provides numbers of model operational commands, including load, store, import and etc.
    For more details, please check out the help text.
    """
    pass


@model.command("load", short_help="Load Model Command")
@cli_args.JOBID
@click.option("-c", "--conf-path", type=click.Path(exists=True),
              help="Configuration file path.")
@click.pass_context
def load(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        Load Model Command

    \b
    - USAGE:
        flow model load -c fate_flow/examples/publish_load_model.json
        flow model load -j $JOB_ID
    """
    if not kwargs.get("conf_path") and not kwargs.get("job_id"):
        prettify(
            {
                "retcode": 100,
                "retmsg": "Load model failed. No arguments received, "
                          "please provide one of arguments from job id and conf path."
            }
        )
    else:
        if kwargs.get("conf_path") and kwargs.get("job_id"):
            prettify(
                {
                    "retcode": 100,
                    "retmsg": "Load model failed. Please do not provide job id and "
                              "conf path at the same time."
                }
            )
        else:
            config_data, dsl_data = preprocess(**kwargs)
            access_server('post', ctx, 'model/load', config_data)


@model.command("bind", short_help="Bind Model Command")
@cli_args.JOBID
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
        flow model bind -c fate_flow/examples/bind_model_service.json -j $JOB_ID
    """
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, 'model/bind', config_data)


@model.command("import", short_help="Import Model Command")
@cli_args.CONF_PATH
# @click.option("--force", is_flag=True, default=False,
#               help="If specified, the existing model file which is named the same as current model's, "
#                    "older version of model files would be renamed. Otherwise, importing progress would be rejected.")
@click.option('--from-database', is_flag=True, default=False,
              help="If specified and there is a valid database environment, fate flow will import model from database "
                   "which you specified in configuration file.")
@click.pass_context
def import_model(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        Import Model Command. Users can currently import models from files or databases (including mysql and redis).

    \b
    - USAGE:
        flow model import -c fate_flow/examples/import_model.json
        flow model import -c fate_flow/examples/restore_model.json --from-database
    """
    config_data, dsl_data = preprocess(**kwargs)
    if not config_data.pop('from_database'):
        file_path = config_data.get("file", None)
        if file_path:
            if not os.path.isabs(file_path):
                file_path = os.path.join(get_project_base_directory(), file_path)
            if os.path.exists(file_path):
                files = {'file': open(file_path, 'rb')}
                access_server('post', ctx, 'model/import', data=config_data, files=files)
            else:
                prettify({'retcode': 100,
                          'retmsg': 'Import model failed. The file is obtained from the fate flow client machine, '
                                    'but it does not exist, please check the path: {}'.format(file_path)})
        else:
            prettify({
                'retcode': 100,
                'retmsg': "Import model failed. Please specify the valid model file path and try again."
            })
    else:
        access_server('post', ctx, 'model/restore', config_data)


@model.command("export", short_help="Export Model Command")
@cli_args.CONF_PATH
@click.option('--to-database', is_flag=True, default=False,
              help="If specified and there is a valid database environment, fate flow will export model to database "
                   "which you specified in configuration file.")
@click.pass_context
def export_model(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        Export Model Command. Users can currently export models to files or databases (including mysql and redis).

    \b
    - USAGE:
        flow model export -c fate_flow/examples/export_model.json
        flow model export -c fate_flow/examplse/store_model.json --to-database
    """
    config_data, dsl_data = preprocess(**kwargs)
    if not config_data.pop('to_database'):
        with closing(access_server('get', ctx, 'model/export', config_data, False, stream=True)) as response:
            if response.status_code == 200:
                archive_file_name = re.findall("filename=(.+)", response.headers["Content-Disposition"])[0]
                os.makedirs(config_data["output_path"], exist_ok=True)
                archive_file_path = os.path.join(config_data["output_path"], archive_file_name)
                with open(archive_file_path, 'wb') as fw:
                    for chunk in response.iter_content(1024):
                        if chunk:
                            fw.write(chunk)
                response_dict = {'retcode': 0,
                                 'file': archive_file_path,
                                 'retmsg': 'download successfully, please check {}'.format(archive_file_path)}
            else:
                response_dict = response.json() if isinstance(response, requests.models.Response) else response.json
        prettify(response_dict)
    else:
        access_server('post', ctx, 'model/store', config_data)


@model.command("migrate", short_help="Migrate Model Command")
@cli_args.CONF_PATH
@click.pass_context
def migrate(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        Migrate Model Command.

    \b
    - USAGE:
        flow model migrate -c fate_flow/examples/migrate_model.json
    """
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, 'model/migrate', config_data)


@model.command("tag-model", short_help="Tag Model Command")
@cli_args.JOBID_REQUIRED
@cli_args.TAG_NAME_REQUIRED
@click.option("--remove", is_flag=True, default=False,
              help="If specified, the name of specified model will be "
                   "removed from the model name list of specified tag")
@click.pass_context
def tag_model(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        Tag Model Command.
        By default, custom can execute this command to tag model. Or custom could
        specify the 'remove' flag to remove the tag from model.

    \b
    - USAGE:
        flow model tag-model -j $JOB_ID -t $TAG_NAME
        flow model tag-model -j $JOB_ID -t $TAG_NAME --remove
    """
    config_data, dsl_data = preprocess(**kwargs)
    if not config_data.pop('remove'):
        access_server('post', ctx, 'model/model_tag/create', config_data)
    else:
        access_server('post', ctx, 'model/model_tag/remove', config_data)


@model.command("tag-list", short_help="List Tags of Model Command")
@cli_args.JOBID_REQUIRED
@click.pass_context
def list_tag(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        List Tags of Model Command.
        Custom can query the model by a valid job id, and get the tag list of the specified model.

    \b
    - USAGE:
        flow model tag-list -j $JOB_ID
    """
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, 'model/model_tag/retrieve', config_data)
