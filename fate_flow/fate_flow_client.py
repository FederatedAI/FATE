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
import sys
import argparse
import json
import os
import tarfile
import traceback
from contextlib import closing

import requests

from arch.api.utils import file_utils
from fate_flow.settings import SERVERS, ROLE, API_VERSION
from fate_flow.utils import detect_utils

server_conf = file_utils.load_json_conf("arch/conf/server_conf.json")
JOB_OPERATE_FUNC = ["submit_job", "stop_job", 'query_job']
JOB_FUNC = ["job_config", "job_log"]
TASK_OPERATE_FUNC = ['query_task']
TRACKING_FUNC = ["component_parameters", "component_metric_all", "component_metrics",
                 "component_output_model", 'component_output_data']
DATA_FUNC = ["download", "upload"]
TABLE_FUNC = ["table_info"]
MODEL_FUNC = ["load", "online", "version"]


def prettify(response, verbose=True):
    if verbose:
        print(json.dumps(response, indent=4))
        print()
    return response


def call_fun(func, config_data, dsl_path, config_path):
    ip = server_conf.get(SERVERS).get(ROLE).get('host')
    http_port = server_conf.get(SERVERS).get(ROLE).get('http.port')
    local_url = "http://{}:{}/{}".format(ip, http_port, API_VERSION)

    if func in JOB_OPERATE_FUNC:
        if func == 'submit_job':
            if not config_path:
                raise Exception('the following arguments are required: {}'.format('runtime conf path'))
            dsl_data = {}
            if dsl_path or config_data.get('job_parameters', {}).get('job_type', '') == 'predict':
                if dsl_path:
                    dsl_path = os.path.abspath(dsl_path)
                    with open(dsl_path, 'r') as f:
                        dsl_data = json.load(f)
            else:
                raise Exception('the following arguments are required: {}'.format('dsl path'))
            post_data = {'job_dsl': dsl_data,
                         'job_runtime_conf': config_data}
        else:
            if func != 'query_job':
                detect_utils.check_config(config=config_data, required_arguments=['job_id'])
            post_data = config_data
        response = requests.post("/".join([local_url, "job", func.rstrip('_job')]), json=post_data)
        if func == 'query_job':
            response = response.json()
            if response['retcode'] == 0:
                for i in range(len(response['data'])):
                    del response['data'][i]['f_runtime_conf']
                    del response['data'][i]['f_dsl']
    elif func in JOB_FUNC:
        if func == 'job_config':
            detect_utils.check_config(config=config_data, required_arguments=['job_id', 'role', 'party_id', 'output_path'])
            response = requests.post("/".join([local_url, func.replace('_', '/')]), json=config_data)
            response_data = response.json()
            if response_data['retcode'] == 0:
                job_id = response_data['data']['job_id']
                download_directory = os.path.join(config_data['output_path'], 'job_{}_config'.format(job_id))
                os.makedirs(download_directory, exist_ok=True)
                for k, v in response_data['data'].items():
                    if k == 'job_id':
                        continue
                    with open('{}/{}.json'.format(download_directory, k), 'w') as fw:
                        json.dump(v, fw, indent=4)
                del response_data['data']['dsl']
                del response_data['data']['runtime_conf']
                response_data['directory'] = download_directory
                response_data['retmsg'] = 'download successfully, please check {} directory'.format(download_directory)
                response = response_data
        elif func == 'job_log':
            detect_utils.check_config(config=config_data, required_arguments=['job_id', 'output_path'])
            with closing(requests.get("/".join([local_url, func.replace('_', '/')]), json=config_data,
                                      stream=True)) as response:
                job_id = config_data['job_id']
                tar_file_name = 'job_{}_log.tar.gz'.format(job_id)
                with open(tar_file_name, 'wb') as fw:
                    for chunk in response.iter_content(1024):
                        if chunk:
                            fw.write(chunk)
                extract_dir = os.path.join(config_data['output_path'], 'job_{}_log'.format(job_id))
                tar = tarfile.open(tar_file_name, "r:gz")
                file_names = tar.getnames()
                for file_name in file_names:
                    tar.extract(file_name, extract_dir)
                tar.close()
                os.remove(tar_file_name)
            response = {'retcode': 0,
                        'directory': extract_dir,
                        'retmsg': 'download successfully, please check {} directory'.format(extract_dir)}
    elif func in TASK_OPERATE_FUNC:
        response = requests.post("/".join([local_url, "job", "task", func.rstrip('_task')]), json=config_data)
    elif func in TRACKING_FUNC:
        detect_utils.check_config(config=config_data,
                                  required_arguments=['job_id', 'component_name', 'role', 'party_id'])
        if func == 'component_output_data':
            detect_utils.check_config(config=config_data, required_arguments=['output_path'])
            tar_file_name = 'job_{}_{}_{}_{}_output_data.tar.gz'.format(config_data['job_id'],
                                                                        config_data['component_name'],
                                                                        config_data['role'],
                                                                        config_data['party_id'])
            extract_dir = os.path.join(config_data['output_path'], tar_file_name.replace('.tar.gz', ''))
            with closing(requests.get("/".join([local_url, "tracking", func.replace('_', '/'), 'download']),
                                      json=config_data,
                                      stream=True)) as res:
                if res.status_code == 200:
                    with open(tar_file_name, 'wb') as fw:
                        for chunk in res.iter_content(1024):
                            if chunk:
                                fw.write(chunk)
                    tar = tarfile.open(tar_file_name, "r:gz")
                    file_names = tar.getnames()
                    for file_name in file_names:
                        tar.extract(file_name, extract_dir)
                    tar.close()
                    os.remove(tar_file_name)
                    response = {'retcode': 0,
                                'directory': extract_dir,
                                'retmsg': 'download successfully, please check {} directory'.format(extract_dir)}
                else:
                    response = res.json()

        else:
            response = requests.post("/".join([local_url, "tracking", func.replace('_', '/')]), json=config_data)
    elif func in DATA_FUNC:
        response = requests.post("/".join([local_url, "data", func]), json=config_data)
    elif func in TABLE_FUNC:
        detect_utils.check_config(config=config_data, required_arguments=['namespace', 'table_name'])
        response = requests.post("/".join([local_url, "table", func]), json=config_data)
    elif func in MODEL_FUNC:
        if func == "version":
            detect_utils.check_config(config=config_data, required_arguments=['namespace'])
        response = requests.post("/".join([local_url, "model", func]), json=config_data)
    return response.json() if isinstance(response, requests.models.Response) else response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=False, type=str, help="runtime conf path")
    parser.add_argument('-d', '--dsl', required=False, type=str, help="dsl path")
    parser.add_argument('-f', '--function', type=str,
                        choices=(
                                DATA_FUNC + MODEL_FUNC + JOB_FUNC + JOB_OPERATE_FUNC + TASK_OPERATE_FUNC + TABLE_FUNC + TRACKING_FUNC),
                        required=True,
                        help="function to call")
    parser.add_argument('-j', '--job_id', required=False, type=str, help="job id")
    parser.add_argument('-p', '--party_id', required=False, type=str, help="party id")
    parser.add_argument('-r', '--role', required=False, type=str, help="role")
    parser.add_argument('-cpn', '--component_name', required=False, type=str, help="component name")
    parser.add_argument('-s', '--status', required=False, type=str, help="status")
    parser.add_argument('-n', '--namespace', required=False, type=str, help="namespace")
    parser.add_argument('-t', '--table_name', required=False, type=str, help="table name")
    parser.add_argument('-w', '--work_mode', required=False, type=int, help="work mode")
    parser.add_argument('-i', '--file', required=False, type=str, help="file")
    parser.add_argument('-o', '--output_path', required=False, type=str, help="output_path")
    try:
        args = parser.parse_args()
        config_data = {}
        dsl_path = args.dsl
        config_path = args.config
        if args.config:
            args.config = os.path.abspath(args.config)
            with open(args.config, 'r') as f:
                config_data = json.load(f)
        config_data.update(dict((k, v) for k, v in vars(args).items() if v is not None))
        if args.party_id or args.role:
            config_data['local'] = config_data.get('local', {})
            if args.party_id:
                config_data['local']['party_id'] = args.party_id
            if args.role:
                config_data['local']['role'] = args.role
        response = call_fun(args.function, config_data, dsl_path, config_path)
    except Exception as e:
        exc_type, exc_value, exc_traceback_obj = sys.exc_info()
        response = {'retcode': 100, 'retmsg': str(e), 'traceback': traceback.format_exception(exc_type, exc_value, exc_traceback_obj)}
    response_dict = prettify(response)
