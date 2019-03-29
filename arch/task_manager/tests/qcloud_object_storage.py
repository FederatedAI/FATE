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
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
import traceback
from bottle import post, run, request
import json
import random
import uuid
import threading
from queue import Queue
import requests
import time

secret_id = 'AKIDMaYQadgAbvcIROyA3jCHzbz9XwkMxERd'
secret_key = 'HKAaoolKi8HCwRXhiV8W9bv3B9xayGrw'
region = 'ap-shanghai'
token = None
scheme = 'https'
bucket = 'jarvistest-1256844776'

send_done_url = 'http://127.0.0.1:9380/v1/data/importOfflineFeature'

job_queue = Queue()


@post('/requestOfflineFeature')
def request_offline_feature():
    job_id = request.json.get('jobId')
    config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key, Token=token, Scheme=scheme)
    client = CosS3Client(config)
    file_name = 'test_feature_%s.csv' % (job_id)
    tags = ['tag%d' % n for n in range(50)]
    with open(file_name, 'w') as fw:
        for li in range(100):
            items = [uuid.uuid1().hex]
            items.extend(random.sample(tags, 30))
            fw.write('%s\n' % '\t'.join(items))

    with open(file_name, 'rb') as fp:
        response = client.put_object(
            Bucket=bucket,
            Body=fp,
            Key=file_name,
            StorageClass='STANDARD',
            EnableMD5=True
        )
    print(response)
    job_queue.put((job_id, file_name))
    return json.dumps({"status": 0, "msg": "success"})


def send_done():
    job_id, file_name = job_queue.get()
    print(job_id, file_name)
    time.sleep(5)
    print('send finish signal')
    response = requests.post(send_done_url, json={'jobId': job_id, 'sourcePath': file_name})
    print(response.json())


if __name__ == '__main__':
    th = threading.Thread(target=send_done)
    th.start()
    run(host='127.0.0.1', port=1234, debug=True)
