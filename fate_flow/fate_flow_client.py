#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import tarfile
import time
import traceback
from contextlib import closing

import requests

from arch.api.utils import file_utils

SERVERS = "servers"
ROLE = "manager"
API_VERSION = "v1"
server_conf = file_utils.load_json_conf("arch/conf/server_conf.json")
JOB_OPERATE_FUNC = ["submit_job", "stop_job", 'query_job']
JOB_FUNC = ["job_config", "job_log"]
TASK_OPERATE_FUNC = ['query_task']
TRACKING_FUNC = ["component_parameters", "component_metric_all", "component_metrics", "component_metric_data",
                 "component_output_model", 'component_output_data']
DATA_FUNC = ["download", "upload"]
TABLE_FUNC = ["table_info"]
MODEL_FUNC = ["load", "online", "version"]


def get_err_result(msg, body):
    return {"retcode": -1,
            "retmsg": msg,
            "created_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "data": body}


def prettify(response, verbose=True):
    if verbose:
        print(json.dumps(response, indent=4))
        print()
    return response


def call_fun(func, dsl_data, config_data):
    IP = server_conf.get(SERVERS).get(ROLE).get('host')
    HTTP_PORT = server_conf.get(SERVERS).get(ROLE).get('http.port')
    LOCAL_URL = "http://{}:{}/{}".format(IP, HTTP_PORT, API_VERSION)

    if func in JOB_OPERATE_FUNC:
        if dsl_data and config_data:
            post_data = {'job_dsl': dsl_data,
                         'job_runtime_conf': config_data}
        else:
            post_data = config_data
        response = requests.post("/".join([LOCAL_URL, "job", func.rstrip('_job')]), json=post_data)
    elif func in JOB_FUNC:
        if func == 'job_config':
            response = requests.post("/".join([LOCAL_URL, func.replace('_', '/')]), json=config_data)
            response_data = response.json()
            if response_data['retcode'] == 0:
                job_id = response_data['data']['job_id']
                os.makedirs('job_{}_config'.format(job_id), exist_ok=True)
                for k, v in response_data['data'].items():
                    if k == 'job_id':
                        continue
                    with open('job_{}_config/{}.json'.format(job_id, k), 'w') as fw:
                        json.dump(v, fw, indent=4)
                del response_data['data']['dsl']
                del response_data['data']['runtime_conf']
                response_data['retmsg'] = 'download successfully, please check job_{}_config directory'.format(job_id)
                response = response_data
        elif func == 'job_log':
            try:
                with closing(requests.get("/".join([LOCAL_URL, func.replace('_', '/')]), json=config_data,
                                          stream=True)) as response:
                    job_id = config_data.get('job_id', '')
                    tar_file_name = 'job_{}_log.tar.gz'.format(job_id)
                    with open(tar_file_name, 'wb') as fw:
                        for chunk in response.iter_content(1024):
                            if chunk:
                                fw.write(chunk)
                    extract_dir = 'job_{}_log'.format(job_id)
                    try:
                        tar = tarfile.open(tar_file_name, "r:gz")
                        file_names = tar.getnames()
                        for file_name in file_names:
                            tar.extract(file_name, extract_dir)
                        tar.close()
                        os.remove(tar_file_name)
                    except Exception as e:
                        print(e)
                response = {'retcode': 0,
                            'retmsg': 'download successfully, please check {} directory'.format(extract_dir)}
            except Exception as e:
                traceback.print_exc(e)
                response = {'retcode': 101, 'retmsg': str(e)}
    elif func in TASK_OPERATE_FUNC:
        response = requests.post("/".join([LOCAL_URL, "job", "task", func.rstrip('_task')]), json=config_data)
    elif func in TRACKING_FUNC:
        response = requests.post("/".join([LOCAL_URL, "tracking", func.replace('_', '/')]), json=config_data)
        if response.json().get('retcode', 100) == 0:
            if func == 'component_output_data':
                response = response.json()
                output_file = 'job_{}_{}_output_data.csv'.format(config_data['job_id'], config_data[
                    'component_name']) if 'output_path' not in config_data else config_data['output_path']
                with open(output_file, 'w') as fw:
                    for line_items in response['data']:
                        fw.write('{}\n'.format('\t'.join(map(lambda x: str(x), line_items))))
                del response['data']
                response['retmsg'] = 'download successfully, please check {}'.format(output_file)
            else:
                response = response.json()['data']
    elif func in DATA_FUNC:
        response = requests.post("/".join([LOCAL_URL, "data", func]), json=config_data)
    elif func in TABLE_FUNC:
        response = requests.post("/".join([LOCAL_URL, "datatable", func]), json=config_data)
    elif func in MODEL_FUNC:
        response = requests.post("/".join([LOCAL_URL, "model", func]), json=config_data)
    try:
        return response.json() if isinstance(response, requests.models.Response) else response
    except Exception as e:
        print(response.text)
        traceback.print_exc()
        return {'retcode': 500, 'msg': str(e)}


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
    parser.add_argument('-s', '--status', required=False, type=str, help="status")
    parser.add_argument('-cpn', '--component_name', required=False, type=str, help="component name")
    parser.add_argument('-n', '--namespace', required=False, type=str, help="namespace")
    parser.add_argument('-t', '--table_name', required=False, type=str, help="table name")
    parser.add_argument('-i', '--file', required=False, type=str, help="file")
    parser.add_argument('-o', '--output_path', required=False, type=str, help="output_path")
    try:
        args = parser.parse_args()
        config_data = {}
        dsl_data = {}
        try:
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
            if args.dsl:
                args.dsl = os.path.abspath(args.dsl)
                with open(args.dsl, 'r') as f:
                    dsl_data = json.load(f)
        except ValueError:
            print('json parse error')
            exit(-102)
        except IOError:
            print("reading config jsonfile error")
            exit(-103)

        response = call_fun(args.function, dsl_data, config_data)
        response_dict = prettify(response)
    except:
        traceback.print_exc()
