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
from datetime import datetime

import click
from contextlib import closing

import requests
from flow_client.flow_cli.utils import cli_args
from flow_client.flow_cli.utils.cli_utils import (prettify, preprocess, download_from_request,
                                                  access_server, check_abs_path)


@click.group(short_help="Component Operations")
@click.pass_context
def component(ctx):
    """
    \b
    Provides numbers of component operational commands, including metrics, parameters and etc.
    For more details, please check out the help text.
    """
    pass


@component.command("list", short_help="List Components Command")
@cli_args.JOBID_REQUIRED
@click.pass_context
def list(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        List components of a specified job.

    \b
    - USAGE:
        flow component list -j $JOB_ID
    """
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, 'tracking/component/list', config_data)


@component.command("metrics", short_help="Component Metrics Command")
@cli_args.JOBID_REQUIRED
@cli_args.ROLE_REQUIRED
@cli_args.PARTYID_REQUIRED
@cli_args.COMPONENT_NAME_REQUIRED
@click.pass_context
def metrics(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        Query the List of Metrics.

    \b
    - USAGE:
        flow component metrics -j $JOB_ID -r host -p 10000 -cpn hetero_feature_binning_0
    """
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, 'tracking/component/metrics', config_data)


@component.command("metric-all", short_help="Component Metric All Command")
@cli_args.JOBID_REQUIRED
@cli_args.ROLE_REQUIRED
@cli_args.PARTYID_REQUIRED
@cli_args.COMPONENT_NAME_REQUIRED
@click.pass_context
def metric_all(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        Query All Metric Data.

    \b
    - USAGE:
        flow component metric-all -j $JOB_ID -r host -p 10000 -cpn hetero_feature_binning_0
    """
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, 'tracking/component/metric/all', config_data)


@component.command("metric-delete", short_help="Delete Metric Command")
@click.option('-d', '--date', type=click.STRING,
              help="An 8-digit valid date, format like 'YYYYMMDD'")
@cli_args.JOBID
@click.pass_context
def metric_delete(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        Delete specified metric.
        If you input both two optional arguments, the 'date' argument will be detected in priority,
        while the job id will be ignored.

    \b
    - USAGE:
        flow component metric-delete -d 20200101
        flow component metric-delete -j $JOB_ID
    """
    config_data, dsl_data = preprocess(**kwargs)
    if config_data.get('date'):
        config_data['model'] = config_data.pop('date')
    access_server('post', ctx, 'tracking/component/metric/delete', config_data)


@component.command("parameters", short_help="Component Parameters Command")
@cli_args.JOBID_REQUIRED
@cli_args.ROLE_REQUIRED
@cli_args.PARTYID_REQUIRED
@cli_args.COMPONENT_NAME_REQUIRED
@click.pass_context
def parameters(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        Query the parameters of a specified component.

    \b
    - USAGE:
        flow component parameters -j $JOB_ID -r host -p 10000 -cpn hetero_feature_binning_0
    """
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, 'tracking/component/parameters', config_data)


@component.command("output-data", short_help="Component Output Data Command")
@cli_args.JOBID_REQUIRED
@cli_args.ROLE_REQUIRED
@cli_args.PARTYID_REQUIRED
@cli_args.COMPONENT_NAME_REQUIRED
@cli_args.OUTPUT_PATH_REQUIRED
@click.option('-l', '--limit', type=click.INT, default=-1,
              help='limit count, defaults is -1 (download all output data)')
@click.pass_context
def output_data(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        Download the Output Data of A Specified Component.

    \b
    - USAGE:
        flow component output-data -j $JOB_ID -r host -p 10000 -cpn hetero_feature_binning_0 --output-path ./examples/
    """
    config_data, dsl_data = preprocess(**kwargs)
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
                res = {'retcode': 0,
                       'directory': os.path.abspath(extract_dir),
                       'retmsg': 'Download successfully, please check {} directory'.format(os.path.abspath(extract_dir))}
            except:
                res = {'retcode': 100,
                       'retmsg': 'Download failed, please check if the parameters are correct.'}
        else:
            try:
                res = response.json() if isinstance(response, requests.models.Response) else response
            except Exception:
                res = {'retcode': 100,
                       'retmsg': 'Download failed, for more details please check logs/fate_flow/fate_flow_stat.log.'}
    prettify(res)


@component.command("output-model", short_help="Component Output Model Command")
@cli_args.JOBID_REQUIRED
@cli_args.ROLE_REQUIRED
@cli_args.PARTYID_REQUIRED
@cli_args.COMPONENT_NAME_REQUIRED
@click.pass_context
def output_model(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        Query the Model of A Speicied Component.

    \b
    - USAGE:
        flow component output-model -j $JOB_ID -r host -p 10000 -cpn hetero_feature_binning_0
    """
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, 'tracking/component/output/model', config_data)


@component.command("output-data-table", short_help="Component Output Data Table Command")
@cli_args.JOBID_REQUIRED
@cli_args.ROLE_REQUIRED
@cli_args.PARTYID_REQUIRED
@cli_args.COMPONENT_NAME_REQUIRED
@click.pass_context
def output_data_table(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        View Table Name and Namespace.

    \b
    - USAGE:
        flow component output-data-table -j $JOB_ID -r host -p 10000 -cpn hetero_feature_binning_0
    """
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, 'tracking/component/output/data/table', config_data)


@component.command("get-summary", short_help="Download Component Summary Command")
@cli_args.JOBID_REQUIRED
@cli_args.ROLE_REQUIRED
@cli_args.PARTYID_REQUIRED
@cli_args.COMPONENT_NAME_REQUIRED
@cli_args.OUTPUT_PATH
@click.pass_context
def download_summary(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        Download summary of a specified component and save it as a json file.

    \b
    - USAGE:
        flow component download-summary -j $JOB_ID -r host -p 10000 -cpn hetero_feature_binning_0
        flow component download-summary -j $JOB_ID -r host -p 10000 -cpn hetero_feature_binning_0 -o ./examples/
    """
    config_data, dsl_data = preprocess(**kwargs)
    if config_data.get("output_path"):
        if not os.path.isdir(config_data.get("output_path")):
            res = {
                "retcode": 100,
                "retmsg": "Please input a valid directory path."
            }
        else:
            config_data["filename"] = "summary_{}_{}.json".format(config_data['component_name'],
                                                                  datetime.now().strftime('%Y%m%d%H%M%S'))
            config_data["output_path"] = os.path.join(check_abs_path(config_data["output_path"]), config_data["filename"])
            with closing(access_server("post", ctx, "tracking/component/summary/download",
                                       config_data, False, stream=True)) as response:
                if response.status_code == 200:
                    with open(config_data["output_path"], "wb") as fout:
                        for chunk in response.iter_content(1024):
                            if chunk:
                                fout.write(chunk)
                    res = {
                        "retcode": 0,
                        "retmsg": "The summary of component <{}> has been stored successfully. "
                                  "File path is: {}.".format(config_data["component_name"],
                                                             config_data["output_path"])
                    }
                else:
                    try:
                        res = response.json() if isinstance(response, requests.models.Response) else response
                    except Exception:
                        res = {"retcode": 100,
                               "retmsg": "Download component summary failed, "
                                         "for more details, please check logs/fate_flow/fate_flow_stat.log."}
        prettify(res)
    else:
        access_server("post", ctx, "tracking/component/summary/download", config_data)
