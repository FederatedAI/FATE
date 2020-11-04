import os
import time
import unittest

import requests
from fate_arch.common import file_utils, conf_utils

from fate_flow.settings import HTTP_PORT, API_VERSION, WORK_MODE, FATEFLOW_SERVICE_NAME
from fate_flow.entity.types import JobStatus

ip = conf_utils.get_base_config(FATEFLOW_SERVICE_NAME).get("host")
server_url = "http://{}:{}/{}".format(ip, HTTP_PORT, API_VERSION)

class TestTable(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.join(file_utils.get_project_base_directory(), "examples", "data")
        self.upload_config = {"file": os.path.join(self.data_dir, "breast_hetero_guest.csv"), "head": 1,
                              "partition": 10, "work_mode": WORK_MODE, "namespace": "fate_flow_test_table_breast_hetero",
                              "table_name": "breast_hetero_guest", "use_local_data": 0, 'drop': 1, 'backend': 0, "id_delimiter": ','}


    def test_upload_guest(self):
        response = requests.post("/".join([server_url, 'data', 'upload']), json=self.upload_config)
        self.assertTrue(response.status_code in [200, 201])
        self.assertTrue(int(response.json()['retcode']) == 0)
        job_id = response.json()['jobId']
        for i in range(60):
            response = requests.post("/".join([server_url, 'job', 'query']), json={'job_id': job_id})
            self.assertTrue(int(response.json()['retcode']) == 0)
            if response.json()['data'][0]['f_status'] == JobStatus.SUCCESS:
                break
            time.sleep(1)
        self.assertTrue(response.json()['data'][0]['f_status'] == JobStatus.SUCCESS)

        response = test_table_info()
        self.assertTrue(response.status_code in [200, 201])
        self.assertTrue(int(response.json()['retcode']) == 0)

        response = test_table_delete()
        self.assertTrue(response.status_code in [200, 201])
        self.assertTrue(int(response.json()['retcode']) == 0)



def test_table_info():
    response = requests.post("/".join([server_url, 'table', 'table_info']),
                             json={'table_name': 'breast_hetero_guest',
                                   'namespace': 'fate_flow_test_table_breast_hetero'})
    return response


def test_table_delete():
    # submit
    response = requests.post("/".join([server_url, 'table', 'delete']),
                             json={'table_name': 'breast_hetero_guest',
                                   'namespace': 'fate_flow_test_table_breast_hetero'})
    return response


if __name__ == '__main__':
    unittest.main()


