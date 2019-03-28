#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import requests
import time
import traceback
from arch.api.utils import file_utils


SERVERS = "servers"
ROLE = "manager"
server_conf = file_utils.load_json_conf("arch/conf/server_conf.json")
WORKFLOW_FUNC = ["train", "predict", "intersect", "cross_validation"]
DATA_FUNC = ["download", "upload"]
OTHER_FUNC = ["delete"]


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


def call_fun(func, data, config_path):
    IP = server_conf.get(SERVERS).get(ROLE).get('host')
    HTTP_PORT = server_conf.get(SERVERS).get(ROLE).get('http.port')
    LOCAL_URL = "http://{}:{}".format(IP, HTTP_PORT)
    print (LOCAL_URL)

    if func in WORKFLOW_FUNC:
        response = requests.post("/".join([LOCAL_URL, "job"]), json=data)
    elif func in OTHER_FUNC:
        response = requests.delete("/".join([LOCAL_URL, "job", data.get("jobid")]))
    elif func in DATA_FUNC:
        print ("enter here", config_path)
        response = requests.post("/".join([LOCAL_URL, "data", func]), json={"config_path": config_path})

    return json.loads(response.text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=False, type=str, help="config json path")
    parser.add_argument('-f', '--function', type=str,
                        choices=WORKFLOW_FUNC + DATA_FUNC + OTHER_FUNC,
                        required=True,
                        help="function to call")
    try:
        args = parser.parse_args()
        if not args.config:
            exit(-100)
        
        data = {}
        try:
            args.config = os.path.abspath(args.config)
            with open(args.config, 'r') as f:
                data = json.load(f)
        except ValueError:
            print('json parse error')
            exit(-102)
        except IOError:
            print("reading config jsonfile error")
            exit(-103)

        response = call_fun(args.function.lower(), data, args.config)

        print('===== Task Submit Result =====\n')
        response_dict = prettify(response)
        if response.get("status") < 0:
            result = get_err_result(response.msg, str(response_dict.get('data')))
            print(result)
            sys.exit(result.get("code"))

    except:
        traceback.print_exc()
