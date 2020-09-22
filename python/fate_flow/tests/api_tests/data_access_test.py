import os
import time
import unittest

import requests
from fate_arch.common import file_utils, conf_utils

from fate_flow.settings import HTTP_PORT, API_VERSION, WORK_MODE, FATEFLOW_SERVICE_NAME


class TestDataAccess(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.join(file_utils.get_project_base_directory(), "examples", "data")
        self.upload_guest_config = {"file": os.path.join(self.data_dir, "breast_hetero_guest.csv"), "head": 1,
                                    "partition": 10, "work_mode": WORK_MODE, "namespace": "fate_flow_test_breast_hetero",
                                    "table_name": "breast_hetero_guest", "use_local_data": 0, 'drop': 0, 'backend': 0, "id_delimiter": ',',}
        self.upload_host_config = {"file": os.path.join(self.data_dir, "breast_hetero_host.csv"), "head": 1,
                                   "partition": 10, "work_mode": WORK_MODE, "namespace": "fate_flow_test_breast_hetero",
                                   "table_name": "breast_hetero_host", "use_local_data": 0, 'drop': 0, 'backend': 0, "id_delimiter": ',',}
        self.download_config = {"output_path": os.path.join(file_utils.get_project_base_directory(),
                                                            "fate_flow/fate_flow_unittest_breast_b.csv"),
                                "work_mode": WORK_MODE, "namespace": "fate_flow_test_breast_hetero",
                                "table_name": "breast_hetero_guest"}
        ip = conf_utils.get_base_config(FATEFLOW_SERVICE_NAME).get("host")
        self.server_url = "http://{}:{}/{}".format(ip, HTTP_PORT, API_VERSION)

    def test_upload_guest(self):
        response = requests.post("/".join([self.server_url, 'data', 'upload']), json=self.upload_guest_config)
        self.assertTrue(response.status_code in [200, 201])
        self.assertTrue(int(response.json()['retcode']) == 0)
        job_id = response.json()['jobId']
        for i in range(60):
            response = requests.post("/".join([self.server_url, 'job', 'query']), json={'job_id': job_id})
            self.assertTrue(int(response.json()['retcode']) == 0)
            if response.json()['data'][0]['f_status'] == 'success':
                break
            time.sleep(1)

    def test_upload_host(self):
        response = requests.post("/".join([self.server_url, 'data', 'upload']), json=self.upload_host_config)
        self.assertTrue(response.status_code in [200, 201])
        self.assertTrue(int(response.json()['retcode']) == 0)
        job_id = response.json()['jobId']
        for i in range(60):
            response = requests.post("/".join([self.server_url, 'job', 'query']), json={'job_id': job_id})
            self.assertTrue(int(response.json()['retcode']) == 0)
            if response.json()['data'][0]['f_status'] == 'success':
                break
            time.sleep(1)

    def test_upload_history(self):
        response = requests.post("/".join([self.server_url, 'data', 'upload/history']), json={'limit': 2})
        self.assertTrue(response.status_code in [200, 201])
        self.assertTrue(int(response.json()['retcode']) == 0)

    def test_download(self):
        response = requests.post("/".join([self.server_url, 'data', 'download']), json=self.download_config)
        self.assertTrue(response.status_code in [200, 201])
        self.assertTrue(int(response.json()['retcode']) == 0)


if __name__ == '__main__':
    unittest.main()


