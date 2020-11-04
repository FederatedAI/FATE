import json
import os
import time
import unittest

import requests
from fate_arch.common import file_utils, conf_utils

from fate_flow.settings import HTTP_PORT, API_VERSION, WORK_MODE, FATEFLOW_SERVICE_NAME
from fate_flow.entity.types import EndStatus, JobStatus


class TestTracking(unittest.TestCase):
    def setUp(self):
        self.sleep_time = 10
        self.success_job_dir = './jobs/'
        self.dsl_path = 'fate_flow/examples/test_hetero_lr_job_dsl.json'
        self.config_path = 'fate_flow/examples/test_hetero_lr_job_conf.json'
        self.test_component_name = 'hetero_feature_selection_0'
        ip = conf_utils.get_base_config(FATEFLOW_SERVICE_NAME).get("host")
        self.server_url = "http://{}:{}/{}".format(ip, HTTP_PORT, API_VERSION)
        self.party_info = file_utils.load_json_conf(os.path.abspath(os.path.join('./jobs', 'party_info.json'))) if WORK_MODE else None
        self.guest_party_id = self.party_info['guest'] if WORK_MODE else 9999
        self.host_party_id = self.party_info['host'] if WORK_MODE else 10000

    def test_tracking(self):
        with open(os.path.join(file_utils.get_python_base_directory(), self.dsl_path), 'r') as f:
            dsl_data = json.load(f)
        with open(os.path.join(file_utils.get_python_base_directory(), self.config_path), 'r') as f:
            config_data = json.load(f)
            config_data['job_parameters']['work_mode'] = WORK_MODE
            config_data[ "initiator"]["party_id"] = self.guest_party_id
            config_data["role"] = {
                "guest": [self.guest_party_id],
                "host": [self.host_party_id],
                "arbiter": [self.host_party_id]
            }
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
            if EndStatus.contains(job_info['f_status']):
                break
            time.sleep(self.sleep_time)
            print('waiting job run success, the job has been running for {}s'.format((i+1)*self.sleep_time))
        self.assertTrue(job_info['f_status'] == JobStatus.SUCCESS)
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


