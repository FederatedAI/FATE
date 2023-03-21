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
from datetime import datetime

import click
import requests
from flow_client.flow_cli.utils import cli_args
from contextlib import closing
from flow_client.flow_cli.utils.cli_utils import preprocess, access_server, prettify, get_project_base_directory, \
    check_abs_path


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
@click.option('--from-database', is_flag=True, default=False,
              help="If specified and there is a valid database environment, fate flow will import model from database "
                   "which you specified in configuration file.")
@click.pass_context
def import_model(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        Import the model from a file or storage engine.

    \b
    - USAGE:
        flow model import -c fate_flow/examples/import_model.json
        flow model import -c fate_flow/examples/restore_model.json --from-database
    """
    config_data, dsl_data = preprocess(**kwargs)

    if config_data.pop('from_database'):
        access_server('post', ctx, 'model/restore', config_data)
        return

    file_path = config_data.get("file", None)
    if not file_path:
        prettify({
            'retcode': 100,
            'retmsg': "Import model failed. Please specify the valid model file path and try again."
        })
        return

    if not os.path.isabs(file_path):
        file_path = os.path.join(get_project_base_directory(), file_path)

    if not os.path.exists(file_path):
        prettify({
            'retcode': 100,
            'retmsg': 'Import model failed. The file is obtained from the fate flow client machine, '
                      'but it does not exist, please check the path: {}'.format(file_path),
        })

    config_data['force_update'] = int(config_data.get('force_update', False))
    files = {'file': open(file_path, 'rb')}
    access_server('post', ctx, 'model/import', data=config_data, files=files)


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
        Export the model to a file or storage engine.

    \b
    - USAGE:
        flow model export -c fate_flow/examples/export_model.json
        flow model export -c fate_flow/examples/store_model.json --to-database
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


@model.command("get-predict-dsl", short_help="Get predict dsl of model")
@cli_args.MODEL_ID_REQUIRED
@cli_args.MODEL_VERSION_REQUIRED
@cli_args.OUTPUT_PATH_REQUIRED
@click.pass_context
def get_predict_dsl(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        Get predict DSL of the model.

    \b
    - USAGE:
        flow model get-predict-dsl --model-id $MODEL_ID --model-version $MODEL_VERSION -o ./examples/
    """
    config_data, dsl_data = preprocess(**kwargs)
    dsl_filename = "predict_dsl_{}.json".format(datetime.now().strftime('%Y%m%d%H%M%S'))
    output_path = os.path.join(check_abs_path(kwargs.get("output_path")), dsl_filename)
    config_data["filename"] = dsl_filename

    with closing(access_server('post', ctx, 'model/get/predict/dsl', config_data, False, stream=True)) as response:
        if response.status_code == 200:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as fw:
                for chunk in response.iter_content(1024):
                    if chunk:
                        fw.write(chunk)
            res = {'retcode': 0,
                   'retmsg': "Query predict dsl successfully. "
                             "File path is: {}".format(output_path)}
        else:
            try:
                res = response.json() if isinstance(response, requests.models.Response) else response
            except Exception:
                res = {'retcode': 100,
                       'retmsg': "Query predict dsl failed."
                                 "For more details, please check logs/fate_flow/fate_flow_stat.log"}
    prettify(res)


@model.command("get-predict-conf", short_help="Get predict conf template")
@cli_args.MODEL_ID_REQUIRED
@cli_args.MODEL_VERSION_REQUIRED
@cli_args.OUTPUT_PATH_REQUIRED
@click.pass_context
def get_predict_conf(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        Get the template of predict config.

    \b
    - USAGE:
        flow model get-predict-conf --model-id $MODEL_ID --model-version $MODEL_VERSION -o ./examples/

    """
    config_data, dsl_data = preprocess(**kwargs)
    conf_filename = "predict_conf_{}.json".format(datetime.now().strftime('%Y%m%d%H%M%S'))
    output_path = os.path.join(check_abs_path(kwargs.get("output_path")), conf_filename)
    config_data["filename"] = conf_filename

    with closing(access_server('post', ctx, 'model/get/predict/conf', config_data, False, stream=True)) as response:
        if response.status_code == 200:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as fw:
                for chunk in response.iter_content(1024):
                    if chunk:
                        fw.write(chunk)
            res = {'retcode': 0,
                   'retmsg': "Query predict conf successfully. "
                             "File path is: {}".format(output_path)}
        else:
            try:
                res = response.json() if isinstance(response, requests.models.Response) else response
            except Exception:
                res = {'retcode': 100,
                       'retmsg': "Query predict conf failed."
                                 "For more details, please check logs/fate_flow/fate_flow_stat.log"}
    prettify(res)


@model.command("deploy", short_help="Deploy model")
@cli_args.MODEL_ID_REQUIRED
@cli_args.MODEL_VERSION_REQUIRED
@click.option("--cpn-list", type=click.STRING,
              help="User inputs a string to specify component list")
@click.option("--cpn-path", type=click.Path(exists=True),
              help="User specifies a file path which records the component list.")
@click.option("--dsl-path", type=click.Path(exists=True),
              help="User specified predict dsl file")
@click.option("--cpn-step-index", type=click.STRING, multiple=True,
              help="Specify a checkpoint model to replace the pipeline model. "
                   "Use : to separate component name and step index (E.g. --cpn-step-index cpn_a:123)")
@click.option("--cpn-step-name", type=click.STRING, multiple=True,
              help="Specify a checkpoint model to replace the pipeline model. "
                   "Use : to separate component name and step name (E.g. --cpn-step-name cpn_b:foobar)")
@click.pass_context
def deploy(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        Deploy model.

    \b
    - USAGE:
        flow model deploy --model-id $MODEL_ID --model-version $MODEL_VERSION

    """
    request_data = {
        'model_id': kwargs['model_id'],
        'model_version': kwargs['model_version'],
    }

    if kwargs.get("cpn_list") or kwargs.get("cpn_path"):
        if kwargs.get("cpn_list"):
            cpn_str = kwargs["cpn_list"]
        elif kwargs.get("cpn_path"):
            with open(kwargs["cpn_path"], "r") as fp:
                cpn_str = fp.read()
        else:
            cpn_str = ""

        if isinstance(cpn_str, list):
            cpn_list = cpn_str
        else:
            if (cpn_str.find("/") and cpn_str.find("\\")) != -1:
                raise Exception("Component list string should not contain '/' or '\\'.")
            cpn_str = cpn_str.replace(" ", "").replace("\n", "").strip(",[]")
            cpn_list = cpn_str.split(",")
        request_data['cpn_list'] = cpn_list
    elif kwargs.get("dsl_path"):
        with open(kwargs["dsl_path"], "r") as ft:
            predict_dsl = ft.read()
        request_data['dsl'] = predict_dsl

    request_data['components_checkpoint'] = {}
    for i in ('cpn_step_index', 'cpn_step_name'):
        for j in kwargs[i]:
            component, checkpoint = j.rsplit(':', 1)

            if i == 'cpn_step_index':
                checkpoint = int(checkpoint)
            if component in request_data['components_checkpoint']:
                raise KeyError(f"Duplicated component name '{component}'.")

            request_data['components_checkpoint'][component] = {
                i[4:]: checkpoint,
            }

    config_data, dsl_data = preprocess(**request_data)
    access_server('post', ctx, 'model/deploy', config_data)


@model.command("get-model-info", short_help="Get model info")
@cli_args.MODEL_ID
@cli_args.MODEL_VERSION_REQUIRED
@cli_args.ROLE
@cli_args.PARTYID
@click.option('--detail', is_flag=True, default=False,
              help="If specified, details of model will be shown.")
@click.pass_context
def get_model_info(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        Get model information.

    \b
    - USAGE:
        flow model model-info --model-id $MODEL_ID --model-version $MODEL_VERSION
        flow model model-info --model-id $MODEL_ID --model-version $MODEL_VERSION --detail
    """
    config_data, dsl_data = preprocess(**kwargs)
    if not config_data.pop('detail'):
        config_data['query_filters'] = ['create_date', 'role', 'party_id', 'roles', 'model_id',
                                        'model_version', 'loaded_times', 'size', 'description', 'parent', 'parent_info']
    access_server('post', ctx, 'model/query', config_data)


@model.command("homo-convert", short_help="Convert trained homogenous model")
@cli_args.CONF_PATH
@click.pass_context
def homo_convert_model(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        Convert trained homogenous model to the format of another ML framework. Converted model files
        will be saved alongside the original model and can be downloaded via model export command.
        The supported conversions are:
          HomoLR to `sklearn.linear_model.LogisticRegression`
          HomoNN to `tf.keras.Sequential` or `torch.nn.Sequential`, depending on the originally-used backend type.

    \b
    - USAGE:
        flow model homo-convert -c fate_flow/examples/homo_convert_model.json
    """
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, 'model/homo/convert', config_data)


@model.command("homo-deploy", short_help="Deploy trained homogenous model")
@cli_args.CONF_PATH
@click.pass_context
def homo_deploy_model(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        Deploy trained homogenous model to a target online serving system. The model must be
        converted beforehand.
        Currently the supported target serving system is KFServing. Refer to the example json
        for detailed parameters.

    \b
    - USAGE:
        flow model homo-deploy -c fate_flow/examples/homo_deploy_model.json
    """
    config_data, dsl_data = preprocess(**kwargs)
    if config_data.get('deployment_type') == "kfserving":
        kube_config = config_data.get('deployment_parameters', {}).get('config_file')
        if kube_config:
            if check_abs_path(kube_config):
                with open(kube_config, 'r') as fp:
                    config_data['deployment_parameters']['config_file_content'] = fp.read()
                del config_data['deployment_parameters']['config_file']
            else:
                prettify(
                    {
                        "retcode": 100,
                        "retmsg": "The kube_config file is obtained from the fate flow client machine, "
                                  "but it does not exist. Please check the path: {}".format(kube_config)
                    }
                )
                return
    access_server('post', ctx, 'model/homo/deploy', config_data)
