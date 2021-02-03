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
import click
import socket
import typing
import tarfile
import requests
import traceback
import configparser
from flask import Response


def check_config(config: typing.Dict, required_arguments: typing.List):
    no_arguments = []
    error_arguments = []
    for require_argument in required_arguments:
        if isinstance(require_argument, tuple):
            config_value = config.get(require_argument[0], None)
            if isinstance(require_argument[1], (tuple, list)):
                if config_value not in require_argument[1]:
                    error_arguments.append(require_argument)
            elif config_value != require_argument[1]:
                error_arguments.append(require_argument)
        elif require_argument not in config:
            no_arguments.append(require_argument)
    if no_arguments or error_arguments:
        raise Exception('the following arguments are required: {} {}'.format(','.join(no_arguments), ','.join(['{}={}'.format(a[0], a[1]) for a in error_arguments])))


def prettify(response, verbose=True):
    if verbose:
        if isinstance(response, requests.models.Response):
            try:
                response_dict = response.json()
            except TypeError:
                response_dict = response.json
        else:
            response_dict = response
        try:
            click.echo(json.dumps(response_dict, indent=4, ensure_ascii=False))
        except TypeError:
            click.echo(json.dumps(response_dict.json, indent=4, ensure_ascii=False))
        click.echo('')
    return response


def access_server(method, ctx, postfix, json_data=None, echo=True, **kwargs):
    if ctx.obj.get('init', False):
        try:
            url = "/".join([ctx.obj['server_url'], postfix])
            response = {}
            if method == 'get':
                response = requests.get(url=url, json=json_data, **kwargs)
            elif method == 'post':
                response = requests.post(url=url, json=json_data, **kwargs)
            if echo:
                prettify(response)
                return
            else:
                return response
        except Exception as e:
            exc_type, exc_value, exc_traceback_obj = sys.exc_info()
            response = {'retcode': 100, 'retmsg': str(e),
                        'traceback': traceback.format_exception(exc_type, exc_value, exc_traceback_obj)}
            if 'Connection refused' in str(e):
                response['retmsg'] = 'Connection refused. Please check if the fate flow service is started'
                del response['traceback']
            if 'Connection aborted' in str(e):
                response['retmsg'] = 'Connection aborted. Please make sure that the address of fate flow server ' \
                                     'is configured correctly. The configuration file path is: ' \
                                     '{}.'.format(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                                  os.pardir, os.pardir, 'settings.yaml')))
                del response['traceback']
            if echo:
                prettify(response)
                return
            else:
                return Response(json.dumps(response), status=500, mimetype='application/json')
    else:
        response = {
                'retcode': 100,
                'retmsg': "Fate flow CLI has not been initialized yet or configured incorrectly. "
                          "Please initialize it before using CLI at the first time. And make sure "
                          "the address of fate flow server is configured correctly. The configuration "
                          "file path is: {}.".format(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                                     os.pardir, os.pardir, 'settings.yaml')))
                }
        if echo:
            prettify(response)
        else:
            return Response(json.dumps(response), status=500, mimetype='application/json')


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


def check_abs_path(path):
    if not os.path.isabs(path):
        return os.path.join(os.path.abspath(os.curdir), path).replace("./", "")
    return path


def get_project_base_directory():
    config = configparser.ConfigParser()
    config.read_file(open(os.path.join(os.path.dirname(__file__), os.pardir, 'settings.ini')))
    return config["fate_root"]["project_path"]


def string_to_bytes(string):
    return string if isinstance(string, bytes) else string.encode(encoding="utf-8")
