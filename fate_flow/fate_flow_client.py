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
import time
import re

import requests

from arch.api.utils import file_utils
from arch.api.utils.core_utils import get_lan_ip
from fate_flow.settings import SERVERS, ROLE, API_VERSION, USE_LOCAL_DATA
from fate_flow.utils import detect_utils

server_conf = file_utils.load_json_conf("arch/conf/server_conf.json")
JOB_OPERATE_FUNC = ["submit_job", "stop_job", "query_job", "data_view_query", "clean_job", "clean_queue"]
JOB_FUNC = ["job_config", "job_log"]
TASK_OPERATE_FUNC = ["query_task"]
TRACKING_FUNC = ["component_parameters", "component_metric_all", "component_metric_delete", "component_metrics",
                 "component_output_model", "component_output_data", "component_output_data_table"]
DATA_FUNC = ["download", "upload", "upload_history"]
TABLE_FUNC = ["table_info", "table_delete"]
MODEL_FUNC = ["load", "bind", "store", "restore", "export", "import"]
PERMISSION_FUNC = ["grant_privilege", "delete_privilege", "query_privilege"]


def prettify(response, verbose=True):
    if verbose:
        print(json.dumps(response, indent=4, ensure_ascii=False))
        print()
    return response


def call_fun(func, config_data, dsl_path, config_path):
    ip = server_conf.get(SERVERS).get(ROLE).get('host')
    if ip in ['localhost', '127.0.0.1']:
        ip = get_lan_ip()
    http_port = server_conf.get(SERVERS).get(ROLE).get('http.port')
    server_url = "http://{}:{}/{}".format(ip, http_port, API_VERSION)

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
            response = requests.post("/".join([server_url, "job", func.rstrip('_job')]), json=post_data)
            try:
                if response.json()['retcode'] == 999:
                    start_cluster_standalone_job_server()
                    response = requests.post("/".join([server_url, "job", func.rstrip('_job')]), json=post_data)
            except:
                pass
        elif func == 'data_view_query' or func == 'clean_queue':
            response = requests.post("/".join([server_url, "job", func.replace('_', '/')]), json=config_data)
        else:
            if func != 'query_job':
                detect_utils.check_config(config=config_data, required_arguments=['job_id'])
            post_data = config_data
            response = requests.post("/".join([server_url, "job", func.rstrip('_job')]), json=post_data)
            if func == 'query_job':
                response = response.json()
                if response['retcode'] == 0:
                    for i in range(len(response['data'])):
                        del response['data'][i]['f_runtime_conf']
                        del response['data'][i]['f_dsl']
    elif func in JOB_FUNC:
        if func == 'job_config':
            detect_utils.check_config(config=config_data, required_arguments=['job_id', 'role', 'party_id', 'output_path'])
            response = requests.post("/".join([server_url, func.replace('_', '/')]), json=config_data)
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
            job_id = config_data['job_id']
            tar_file_name = 'job_{}_log.tar.gz'.format(job_id)
            extract_dir = os.path.join(config_data['output_path'], 'job_{}_log'.format(job_id))
            with closing(requests.get("/".join([server_url, func.replace('_', '/')]), json=config_data,
                                      stream=True)) as response:
                if response.status_code == 200:
                    download_from_request(http_response=response, tar_file_name=tar_file_name, extract_dir=extract_dir)
                    response = {'retcode': 0,
                                'directory': extract_dir,
                                'retmsg': 'download successfully, please check {} directory'.format(extract_dir)}
                else:
                    response = response.json()
    elif func in TASK_OPERATE_FUNC:
        response = requests.post("/".join([server_url, "job", "task", func.rstrip('_task')]), json=config_data)
    elif func in TRACKING_FUNC:
        if func != 'component_metric_delete':
            detect_utils.check_config(config=config_data,
                                      required_arguments=['job_id', 'component_name', 'role', 'party_id'])
        if func == 'component_output_data':
            detect_utils.check_config(config=config_data, required_arguments=['output_path'])
            tar_file_name = 'job_{}_{}_{}_{}_output_data.tar.gz'.format(config_data['job_id'],
                                                                        config_data['component_name'],
                                                                        config_data['role'],
                                                                        config_data['party_id'])
            extract_dir = os.path.join(config_data['output_path'], tar_file_name.replace('.tar.gz', ''))
            with closing(requests.get("/".join([server_url, "tracking", func.replace('_', '/'), 'download']),
                                      json=config_data,
                                      stream=True)) as response:
                if response.status_code == 200:
                    try:
                        download_from_request(http_response=response, tar_file_name=tar_file_name, extract_dir=extract_dir)
                        response = {'retcode': 0,
                                    'directory': extract_dir,
                                    'retmsg': 'download successfully, please check {} directory'.format(extract_dir)}
                    except:
                        response = {'retcode': 100,
                                    'retmsg': 'download failed, please check if the parameters are correct'}
                else:
                    response = response.json()

        else:
            response = requests.post("/".join([server_url, "tracking", func.replace('_', '/')]), json=config_data)
    elif func in DATA_FUNC:
        if USE_LOCAL_DATA and func == 'upload':
            file_name = config_data.get('file')
            if not os.path.isabs(file_name):
                file_name = os.path.join(file_utils.get_project_base_directory(), file_name)
            if os.path.exists(file_name):
                files = {'file': open(file_name, 'rb')}
            else:
                raise Exception('The file is obtained from the fate flow client machine, but it does not exist, '
                                'please check the path: {}'.format(file_name))
            response = requests.post("/".join([server_url, "data", func.replace('_', '/')]), data=config_data, files=files)
        else:
            response = requests.post("/".join([server_url, "data", func.replace('_', '/')]), json=config_data)
        try:
            if response.json()['retcode'] == 999:
                start_cluster_standalone_job_server()
                response = requests.post("/".join([server_url, "data", func]), json=config_data)
        except:
            pass
    elif func in TABLE_FUNC:
        if func == "table_info":
            detect_utils.check_config(config=config_data, required_arguments=['namespace', 'table_name'])
            response = requests.post("/".join([server_url, "table", func]), json=config_data)
        else:
            response = requests.post("/".join([server_url, "table", func.lstrip('table_')]), json=config_data)
    elif func in MODEL_FUNC:
        if func == "import":
            file_path = config_data["file"]
            if not os.path.isabs(file_path):
                file_path = os.path.join(file_utils.get_project_base_directory(), file_path)
            if os.path.exists(file_path):
                files = {'file': open(file_path, 'rb')}
            else:
                raise Exception('The file is obtained from the fate flow client machine, but it does not exist, '
                                'please check the path: {}'.format(file_path))
            response = requests.post("/".join([server_url, "model", func]), data=config_data, files=files)
        elif func == "export":
            with closing(requests.get("/".join([server_url, "model", func]), json=config_data, stream=True)) as response:
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
        else:
            response = requests.post("/".join([server_url, "model", func]), json=config_data)
    elif func in PERMISSION_FUNC:
        detect_utils.check_config(config=config_data, required_arguments=['src_party_id', 'src_role'])
        response = requests.post("/".join([server_url, "permission", func.replace('_', '/')]), json=config_data)
    return response.json() if isinstance(response, requests.models.Response) else response


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
    print('use service.sh to start standalone node server....')
    os.system('sh service.sh start --standalone_node')
    time.sleep(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=False, type=str, help="runtime conf path")
    parser.add_argument('-d', '--dsl', required=False, type=str, help="dsl path")
    parser.add_argument('-f', '--function', type=str,
                        choices=(
                                DATA_FUNC + MODEL_FUNC + JOB_FUNC + JOB_OPERATE_FUNC + TASK_OPERATE_FUNC + TABLE_FUNC +
                                TRACKING_FUNC + PERMISSION_FUNC),
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
    parser.add_argument('-m', '--model', required=False, type=str, help="TrackingMetric model id")
    parser.add_argument('-drop', '--drop', required=False, type=str, help="drop data table")
    parser.add_argument('-limit', '--limit', required=False, type=int, help="limit number")
    parser.add_argument('-src_party_id', '--src_party_id', required=False, type=str, help="src party id")
    parser.add_argument('-src_role', '--src_role', required=False, type=str, help="src role")
    parser.add_argument('-privilege_role', '--privilege_role', required=False, type=str, help="privilege role")
    parser.add_argument('-privilege_command', '--privilege_command', required=False, type=str, help="privilege command")
    parser.add_argument('-privilege_component', '--privilege_component', required=False, type=str, help="privilege component")
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
        if config_data.get('output_path'):
            config_data['output_path'] = os.path.abspath(config_data["output_path"])
        response = call_fun(args.function, config_data, dsl_path, config_path)
    except Exception as e:
        exc_type, exc_value, exc_traceback_obj = sys.exc_info()
        response = {'retcode': 100, 'retmsg': str(e), 'traceback': traceback.format_exception(exc_type, exc_value, exc_traceback_obj)}
        if 'Connection refused' in str(e):
            response['retmsg'] = 'Connection refused, Please check if the fate flow service is started'
            del response['traceback']
    response_dict = prettify(response)
