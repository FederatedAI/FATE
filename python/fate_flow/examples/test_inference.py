#  -*-coding:utf8 -*-
import json
import requests
import time
import uuid
import datetime
import time
import argparse

def inference(ids, ip, port):
    url1 = f"http://{ip}:{port}/federation/1.0/inference"
    for id in ids:
        request_data_tmp = {
            "head": {
                "serviceId": "test_model_service",
                "applyId": "209090900991",
            },
            "body": {
                  "featureData": {
                      "phone_num": id,
                  },
                  "sendToRemoteFeatureData": {
                      "device_type": "imei",
                      "phone_num": id,
                      "encrypt_type": "raw"
                  }
            }
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(url1, json=request_data_tmp, headers=headers)
        print("url地址:", url1)
        print("请求信息:\n", request_data_tmp)
        print()
        print("响应信息:\n", response.text)
        print()
        #time.sleep(0.1)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--ids", type=str, nargs='+', help="path of pipeline files", required=False,
                            default=["123", "456"])
    arg_parser.add_argument("--ip", type=str, help="test template", required=False, default="127.0.0.1")
    arg_parser.add_argument("--port", type=str, help="test template", required=False, default="8059")
    args = arg_parser.parse_args()
    ids = args.ids
    ip = args.ip
    port = args.port
    inference(ids=ids, ip=ip, port=port)