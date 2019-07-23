#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import requests
import time
import traceback
from arch.api.utils import file_utils
import sys


SERVERS = "servers"
ROLE = "manager"
API_VERSION = "v1"
server_conf = file_utils.load_json_conf("arch/conf/server_conf.json")
JOB_FUNC = ["submitJob", "jobStatus"]
JOB_QUEUE_FUNC = ["queueStatus"]
DATA_FUNC = ["download", "upload"]
DTABLE_FUNC = ["tableInfo"]
OTHER_FUNC = ["delete"]
MODEL_FUNC = ["load", "online", "version"]


def get_err_result(msg, body):
    return {"retcode": -1,
            "retmsg": msg,
            "created_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "data": body}


def prettify(response, verbose=True):
    response['created_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    if verbose:
        print(json.dumps(response, indent=4))
        print()
    return response


def call_fun(func, dsl_data, config_data):
    IP = server_conf.get(SERVERS).get(ROLE).get('host')
    HTTP_PORT = server_conf.get(SERVERS).get(ROLE).get('http.port')
    LOCAL_URL = "http://{}:{}/{}".format(IP, HTTP_PORT, API_VERSION)

    if func in JOB_FUNC:
        if dsl_data and config_data:
            post_data = {'job_dsl': dsl_data,
                         'job_runtime_conf': config_data}
        else:
            post_data = config_data
        response = requests.post("/".join([LOCAL_URL, "job", func.lstrip('job').rstrip('Job')]), json=post_data)
    elif func in DATA_FUNC:
        response = requests.post("/".join([LOCAL_URL, "data", func]), json=config_data)
    elif func in DTABLE_FUNC:
        response = requests.post("/".join([LOCAL_URL, "datatable", func]), json=config_data)
    elif func in MODEL_FUNC:
        response = requests.post("/".join([LOCAL_URL, "model", func]), json=config_data)

    return json.loads(response.text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=False, type=str, help="runtime conf path")
    parser.add_argument('-d', '--dsl', required=False, type=str, help="dsl path")
    parser.add_argument('-f', '--function', type=str,
                        choices=(DATA_FUNC + OTHER_FUNC + MODEL_FUNC + JOB_FUNC + JOB_QUEUE_FUNC + DATA_FUNC + DTABLE_FUNC),
                        required=True,
                        help="function to call")
    parser.add_argument('-j', '--job_id', required=False, type=str, help="job id")
    parser.add_argument('-p', '--party_id', required=False, type=str, help="party id")
    parser.add_argument('-r', '--role', required=False, type=str, help="role")
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
        if response.get("retcode") < 0:
            result = get_err_result(response_dict.get("retmsg"), response_dict.get('data'))
            sys.exit(result.get("retcode"))

    except:
        traceback.print_exc()
