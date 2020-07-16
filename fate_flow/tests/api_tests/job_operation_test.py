import json
import os
import time
import unittest
from contextlib import closing

import requests
from arch.api.utils import file_utils

from fate_flow.settings import HTTP_PORT, API_VERSION, WORK_MODE


class TestJobOperation(unittest.TestCase):
    def setUp(self):
        self.dsl_path = 'fate_flow/examples/test_hetero_lr_job_dsl.json'
        self.config_path = 'fate_flow/examples/test_hetero_lr_job_conf.json'
        self.server_url = "http://{}:{}/{}".format('127.0.0.1', HTTP_PORT, API_VERSION)

    def test_job_operation(self):
        # submit
        with open(os.path.join(file_utils.get_project_base_directory(), self.dsl_path), 'r') as f:
            dsl_data = json.load(f)
        with open(os.path.join(file_utils.get_project_base_directory(), self.config_path), 'r') as f:
            config_data = json.load(f)
            config_data['job_parameters']['work_mode'] = WORK_MODE
        response = requests.post("/".join([self.server_url, 'job', 'submit']),
                                 json={'job_dsl': dsl_data, 'job_runtime_conf': config_data})
        self.assertTrue(response.status_code in [200, 201])
        self.assertTrue(int(response.json()['retcode']) == 0)

        job_id = response.json()['jobId']

        # query
        response = requests.post("/".join([self.server_url, 'job', 'query']), json={'job_id': job_id, 'role': 'guest'})
        self.assertTrue(int(response.json()['retcode']) == 0)
        job_info = response.json()['data'][0]

        # note
        response = requests.post("/".join([self.server_url, 'job', 'update']), json={'job_id': job_id, 'role': job_info['f_role'], 'party_id': job_info['f_party_id'], 'notes': 'unittest'})
        self.assertTrue(int(response.json()['retcode']) == 0)

        # config
        response = requests.post("/".join([self.server_url, 'job', 'config']), json={'job_id': job_id, 'role': job_info['f_role'], 'party_id': job_info['f_party_id']})
        self.assertTrue(int(response.json()['retcode']) == 0)

        print('waiting 10s ...')
        time.sleep(10)

        # stop
        response = requests.post("/".join([self.server_url, 'job', 'stop']), json={'job_id': job_id})
        self.assertTrue(int(response.json()['retcode']) == 0)

        # logs
        with closing(requests.get("/".join([self.server_url, 'job', 'log']), json={'job_id': job_id}, stream=True)) as response:
            self.assertTrue(response.status_code in [200, 201])

        # query task
        response = requests.post("/".join([self.server_url, 'job', '/task/query']), json={'job_id': job_id})
        self.assertTrue(int(response.json()['retcode']) == 0)

        # query data viw
        response = requests.post("/".join([self.server_url, 'job', '/data/view/query']), json={'party_id': 0})
        self.assertTrue(int(response.json()['retcode']) == 0)

        print('waiting 10s...')
        time.sleep(10)
        # clean job
        response = requests.post("/".join([self.server_url, 'job', '/clean']), json={'job_id': job_id})
        self.assertTrue(int(response.json()['retcode']) == 0)


if __name__ == '__main__':
    unittest.main()


