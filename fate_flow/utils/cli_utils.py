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
import sys
import json
import time
import click
import tarfile
import requests
import traceback


def prettify(response, verbose=True):
    if verbose:
        click.echo(json.dumps(response, indent=4, ensure_ascii=False))
        click.echo('')
    return response


def access_server(method, ctx, postfix, json=None, echo=True, **kwargs):
    try:
        url = "/".join([ctx.obj['server_url'], postfix])
        response = {}
        if method == 'get':
            response = requests.get(url=url, json=json, **kwargs)
        elif method == 'post':
            response = requests.post(url=url, json=json, **kwargs)
        if echo:
            prettify(response.json() if isinstance(response, requests.models.Response) else response)
            return
        else:
            return response
    except Exception as e:
        exc_type, exc_value, exc_traceback_obj = sys.exc_info()
        response = {'retcode': 100, 'retmsg': str(e),
                    'traceback': traceback.format_exception(exc_type, exc_value, exc_traceback_obj)}
        if 'Connection refused' in str(e):
            response['retmsg'] = 'Connection refused, Please check if the fate flow service is started'
            del response['traceback']
        if echo:
            prettify(response.json() if isinstance(response, requests.models.Response) else response)
            return
        else:
            return response


def preprocess(**kwargs):
    config_data = {}

    if kwargs.get('conf_path'):
        conf_path = os.path.abspath(kwargs.get('conf_path'))
        with open(conf_path, 'r') as conf_fp:
            config_data = json.load(conf_fp)

        if config_data.get('output_path'):
            config_data['output_path'] = os.path.abspath(config_data['output_path'])

        if ('party_id' in kwargs.keys()) or ('role' in kwargs.keys()):
            config_data['local'] = config_data.get('local', {})
            if kwargs.get('party_id'):
                config_data['local']['party_id'] = kwargs.get('party_id')
            if kwargs.get('role'):
                config_data['local']['role'] = kwargs.get('role')

    config_data.update(dict((k, v) for k, v in kwargs.items() if v is not None))

    # TODO what if job type is 'predict'
    dsl_data = {}
    if kwargs.get('dsl_path'):
        dsl_path = os.path.abspath(kwargs.get('dsl_path'))
        with open(dsl_path, 'r') as dsl_fp:
            dsl_data = json.load(dsl_fp)
    return config_data, dsl_data


def download_from_request(http_response, tar_file_name, extract_dir):
    with open(tar_file_name, 'wb') as fw:
        for chunk in http_response.iter_content(1024):
            if chunk:
                fw.write(chunk)
    tar = tarfile.open(tar_file_name, "r:gz")
    file_names = tar.getnames()
    for file_name in file_names:
        tar.extract(file_name, extract_dir)
    tar.close()
    os.remove(tar_file_name)


def start_cluster_standalone_job_server():
    click.echo('use service.sh to start standalone node server....')
    os.system('sh service.sh start --standalone_node')
    time.sleep(5)
