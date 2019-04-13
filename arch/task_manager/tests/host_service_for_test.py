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
from arch.api.utils.core import string_to_bytes
from bottle import post, run, request
import json
import random
import uuid
import threading
from queue import Queue
import requests


job_queue = Queue()
tags = ['tag%d' % n for n in range(50)]


def gen_one_feature(id=None):
    items = [uuid.uuid1().hex if not id else id]
    items.extend(random.sample(tags, 30))
    return '%s\n' % '\t'.join(items)


@post('/requestOfflineFeature')
def request_offline_feature():
    job_id = request.json.get('jobId')
    job_queue.put(job_id)
    return json.dumps({"status": 0, "msg": "success"})


@post('/feature')
def request_offline_feature():
    ids = request.json.get('id').split(',')
    data = [gen_one_feature(id) for id in ids]
    response = {"status": 0, "msg": "success", "data": data}
    return response


def send_offline_feature():
    while True:
        job_id = job_queue.get()
        print("get job {}".format(job_id))
        config = CosConfig(Region=settings.region, SecretId=settings.secret_id, SecretKey=settings.secret_key, Token=settings.token, Scheme=settings.scheme)
        client = CosS3Client(config)
        file_name = 'test_feature_%s.csv' % (job_id)
        print("start upload to cos")
        for li in range(100):
            response = client.put_object(
                Bucket=settings.bucket,
                Body=string_to_bytes(gen_one_feature()),
                Key=file_name,
                StorageClass='STANDARD'
            )
        print(response)
        print("upload to cos success")
        print('start to send finish signal')
        response = requests.post(settings.send_done_url, json={'jobId': job_id, 'sourcePath': file_name})
        print('success send finish signal')
        print(response.json())


if __name__ == '__main__':
    th = threading.Thread(target=send_offline_feature)
    th.start()
    run(host='127.0.0.1', port=1234, debug=True)
