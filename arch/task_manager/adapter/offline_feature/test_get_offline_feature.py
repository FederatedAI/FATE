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
import requests
import json
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
from arch.api.utils.core import bytes_to_string
from arch.api.utils.format_transform import list_feature_to_fate_str
import sys
import traceback
REQUEST_OFFLINE_URL = "http://127.0.0.1:1234/requestOfflineFeature"

secret_id = 'AKIDMaYQadgAbvcIROyA3jCHzbz9XwkMxERd'
secret_key = 'HKAaoolKi8HCwRXhiV8W9bv3B9xayGrw'
region = 'ap-shanghai'
token = None
scheme = 'https'
bucket = 'jarvistest-1256844776'


class TestGetOfflineFeature(object):
    @staticmethod
    def request(job_id):
        request_data = {"jobId": job_id}
        response = requests.post(REQUEST_OFFLINE_URL, json=request_data)
        print(response.text)
        return json.loads(response.text)

    @staticmethod
    def import_data(request_data):
        config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key, Token=token, Scheme=scheme)
        client = CosS3Client(config)
        file_name = request_data.get('sourcePath')
        response = client.get_object(
            Bucket=bucket,
            Key=file_name
        )
        fp = response['Body'].get_raw_stream()
        while True:
            line = bytes_to_string(fp.readline())
            if not line:
                break
            values = line.rstrip('\n').split('\t')
            yield (values[0], list_feature_to_fate_str(values[1:]))
