#  -*-coding:utf8 -*-
import json
import requests
import time
import uuid
import datetime
import time

ids = ["18576635456", "13512345432"]

url1 = "http://127.0.0.1:8059/federation/1.0/inference"

for i in range(2):
    request_data_tmp = {
        "head": {
            "serviceId": "test_model_service",
            "applyId": "209090900991",
        },
        "body": {
              "featureData": {
                  "phone_num": ids[i],
              },
              "sendToRemoteFeatureData": {
                  "device_type": "imei",
                  "phone_num": ids[i],
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
