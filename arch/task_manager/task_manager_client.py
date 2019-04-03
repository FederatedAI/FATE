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
server_conf = file_utils.load_json_conf("arch/conf/server_conf.json")
WORKFLOW_FUNC = ["workflow"]
DATA_FUNC = ["download", "upload"]
OTHER_FUNC = ["delete"]
MODEL_FUNC = ["load", "online", "version"]
LOCAL_PROCESS_FUNC = ["import_id", "request_offline_feature"]


def get_err_result(msg, body):
    if not body:
        body = ''
    return {"code": -1,
            "msg": msg,
            "created_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "data": bytes(body, "utf-8")}


def prettify(response, verbose=True):
    data = {"code": response.get("status"), "msg": response.get("msg"), 'created_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    if response.get("data"):
        data["data"] = response.get("data").decode('utf-8')
    if verbose:
        print(data)
    return data


def call_fun(func, data, config_path, input_args):
    print(func)
    IP = server_conf.get(SERVERS).get(ROLE).get('host')
    HTTP_PORT = server_conf.get(SERVERS).get(ROLE).get('http.port')
    LOCAL_URL = "http://{}:{}".format(IP, HTTP_PORT)
    print(LOCAL_URL)

    if func in WORKFLOW_FUNC:
        response = requests.post("/".join([LOCAL_URL, "job", "new"]), json=data)
    elif func in OTHER_FUNC:
        response = requests.delete("/".join([LOCAL_URL, "job", data.get("job_id") or input_args.job_id]))
    elif func in DATA_FUNC:
        print ("enter here", config_path)
        response = requests.post("/".join([LOCAL_URL, "data", func]), json={"config_path": config_path})
    elif func in MODEL_FUNC:
        response = requests.post("/".join([LOCAL_URL, "model", func]), json={"config_path": config_path})
    elif func in LOCAL_PROCESS_FUNC:
        response = eval(func)(LOCAL_URL, data)

    return json.loads(response.text)


def import_id(local_url, config):
    input_file_path = config.request("input_file_path")
    batch_size = config.request("batch_size", 10)
    request_data = {"workMode": config.request("work_mode")}
    with open(input_file_path) as fr:
        id_tmp = []
        range_start = 0
        range_end = -1
        total = 0
        file_end = False
        while True:
            for i in range(batch_size):
                line = fr.readline()
                if not line:
                    file_end = True
                    break
                id_tmp.append(line.split(",")[0])
                range_end += 1
                total += 1
            request_data["rangeStart"] = range_start
            request_data["rangeEnd"] = range_end
            request_data["ids"] = id_tmp
            if file_end:
                # file end
                request_data["total"] = total
                response = requests.post("/".join([local_url, "data/importId"]), json=request_data)
                break
            else:
                request_data["total"] = 0
                response = requests.post("/".join([local_url, "data/importId"]), json=request_data)
            range_start = range_end + 1
            del id_tmp[:]
    return response


def request_offline_feature(local_url, config):
    response = requests.post("/".join([local_url, "data/requestOfflineFeature"]), json=config)
    print(response)
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=False, type=str, help="config json path")
    parser.add_argument('-f', '--function', type=str,
                        choices=WORKFLOW_FUNC + DATA_FUNC + OTHER_FUNC + LOCAL_PROCESS_FUNC + MODEL_FUNC,
                        required=True,
                        help="function to call")
    parser.add_argument('-j', '--job_id', required=False, type=str, help="job id")
    parser.add_argument('-np', '--namespace', required=False, type=str, help="namespace")
    try:
        args = parser.parse_args()
        data = {}
        try:
            if args.config:
                args.config = os.path.abspath(args.config)
                with open(args.config, 'r') as f:
                    data = json.load(f)
        except ValueError:
            print('json parse error')
            exit(-102)
        except IOError:
            print("reading config jsonfile error")
            exit(-103)

        response = call_fun(args.function.lower(), data, args.config, args)

        print('===== Task Submit Result =====\n')
        response_dict = prettify(response)
        if response.get("status") < 0:
            result = get_err_result(response.msg, str(response_dict.get('data')))
            print(result)
            sys.exit(result.get("code"))

    except:
        traceback.print_exc()
