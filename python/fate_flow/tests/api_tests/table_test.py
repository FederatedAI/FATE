import os
import time
import unittest

import requests
from fate_arch.common import file_utils

from fate_flow.settings import HTTP_PORT, API_VERSION, WORK_MODE


class TestTable(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.join(file_utils.get_project_base_directory(), "examples", "data")
        self.upload_config = {"file": os.path.join(self.data_dir, "breast_hetero_guest.csv"), "head": 1, "partition": 10,
                              "work_mode": WORK_MODE, "namespace": "fate_flow_test_table_breast_hetero",
                              "table_name": "breast_hetero_guest", "use_local_data": 0, 'drop': 0}
        self.server_url = "http://{}:{}/{}".format('127.0.0.1', HTTP_PORT, API_VERSION)

    def test_upload_guest(self):
        response = requests.post("/".join([self.server_url, 'data', 'upload']), json=self.upload_config)
        self.assertTrue(response.status_code in [200, 201])
        self.assertTrue(int(response.json()['retcode']) == 0)
        job_id = response.json()['jobId']
        for i in range(60):
            response = requests.post("/".join([self.server_url, 'job', 'query']), json={'job_id': job_id})
            self.assertTrue(int(response.json()['retcode']) == 0)
            if response.json()['data'][0]['f_status'] == 'success':
                break
            time.sleep(1)

    def test_table_info(self):
        response = requests.post("/".join([self.server_url, 'table', 'table_info']),
                                 json={'table_name': 'breast_hetero_guest',
                                       'namespace': 'fate_flow_test_table_breast_hetero'})
        self.assertTrue(response.status_code in [200, 201])
        self.assertTrue(int(response.json()['retcode']) == 0)

    def test_table_delete(self):
        # submit
        response = requests.post("/".join([self.server_url, 'table', 'delete']),
                                 json={'table_name': 'breast_hetero_guest',
                                       'namespace': 'fate_flow_test_table_breast_hetero'})
        self.assertTrue(response.status_code in [200, 201])
        self.assertTrue(int(response.json()['retcode']) == 0)


if __name__ == '__main__':
    unittest.main()


