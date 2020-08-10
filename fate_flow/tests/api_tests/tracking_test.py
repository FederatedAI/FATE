import json
import os
import time
import unittest

import requests
from arch.api.utils import file_utils

from fate_flow.settings import HTTP_PORT, API_VERSION, WORK_MODE


class TestTracking(unittest.TestCase):
    def setUp(self):
        self.sleep_time = 10
        self.success_job_dir = './jobs/'
        self.dsl_path = 'fate_flow/examples/test_hetero_lr_job_dsl.json'
        self.config_path = 'fate_flow/examples/test_hetero_lr_job_conf.json'
        self.test_component_name = 'hetero_feature_selection_0'
        self.server_url = "http://{}:{}/{}".format('127.0.0.1', HTTP_PORT, API_VERSION)

    def test_tracking(self):
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
        job_info = {'f_status': 'running'}
        for i in range(60):
            response = requests.post("/".join([self.server_url, 'job', 'query']), json={'job_id': job_id, 'role': 'guest'})
            self.assertTrue(response.status_code in [200, 201])
            job_info = response.json()['data'][0]
            if job_info['f_status'] in ['success', 'failed', 'canceled']:
                break
            time.sleep(self.sleep_time)
            print('waiting job run success, the job has been running for {}s'.format((i+1)*self.sleep_time))
        self.assertTrue(job_info['f_status'] == 'success')
        os.makedirs(self.success_job_dir, exist_ok=True)
        with open(os.path.join(self.success_job_dir, job_id), 'w') as fw:
            json.dump(job_info, fw)
        self.assertTrue(os.path.exists(os.path.join(self.success_job_dir, job_id)))

        # test_component_parameters
        test_component(self, 'component/parameters')

        # test_component_metric_all
        test_component(self, 'component/metric/all')

        # test_component_metric
        test_component(self, 'component/metrics')

        # test_component_output_model
        test_component(self, 'component/output/model')

        # test_component_output_data_download
        test_component(self, 'component/output/data')

        # test_component_output_data_download
        test_component(self, 'component/output/data/download')

        # test_job_data_view
        test_component(self, 'job/data_view')


def test_component(self, fun):
    job_id = os.listdir(os.path.abspath(os.path.join(self.success_job_dir)))[-1]
    job_info = file_utils.load_json_conf(os.path.abspath(os.path.join(self.success_job_dir, job_id)))
    data = {'job_id': job_id, 'role': job_info['f_role'], 'party_id': job_info['f_party_id'], 'component_name': self.test_component_name}
    if 'download' in fun:
        response = requests.get("/".join([self.server_url, "tracking", fun]), json=data, stream=True)
        self.assertTrue(response.status_code in [200, 201])
    else:
        response = requests.post("/".join([self.server_url, 'tracking', fun]), json=data)
        self.assertTrue(response.status_code in [200, 201])
        self.assertTrue(int(response.json()['retcode']) == 0)


if __name__ == '__main__':
    unittest.main()


