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
import os
from contextlib import closing
from flow_sdk.client.api.base import BaseFlowAPI
from flow_sdk.utils import preprocess, check_config, download_from_request


class Component(BaseFlowAPI):
    def list(self, job_id):
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        return self._post(url='tracking/component/list', json=config_data)

    def metrics(self, job_id, role, party_id, component_name):
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        check_config(config=config_data,
                     required_arguments=['job_id', 'component_name', 'role', 'party_id'])
        return self._post(url='tracking/component/metrics', json=config_data)

    def metric_all(self, job_id, role, party_id, component_name):
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        check_config(config=config_data,
                     required_arguments=['job_id', 'component_name', 'role', 'party_id'])
        return self._post(url='tracking/component/metric/all', json=config_data)

    def metric_delete(self, date=None, job_id=None):
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        if config_data.get('date'):
            config_data['model'] = config_data.pop('date')
        return self._post(url='tracking/component/metric/delete', json=config_data)

    def parameters(self, job_id, role, party_id, component_name):
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        check_config(config=config_data,
                     required_arguments=['job_id', 'component_name', 'role', 'party_id'])
        return self._post(url='tracking/component/parameters', json=config_data)

    def output_data(self, job_id, role, party_id, component_name, output_path, limit=-1):
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        check_config(config=config_data,
                     required_arguments=['job_id', 'component_name', 'role', 'party_id', 'output_path'])
        tar_file_name = 'job_{}_{}_{}_{}_output_data.tar.gz'.format(config_data['job_id'],
                                                                    config_data['component_name'],
                                                                    config_data['role'],
                                                                    config_data['party_id'])
        extract_dir = os.path.join(config_data['output_path'], tar_file_name.replace('.tar.gz', ''))
        with closing(self._get(url='tracking/component/output/data/download',
                               handle_result=False, json=config_data, stream=True)) as response:
            if response.status_code == 200:
                try:
                    download_from_request(http_response=response, tar_file_name=tar_file_name, extract_dir=extract_dir)
                    response = {'retcode': 0,
                                'directory': extract_dir,
                                'retmsg': 'download successfully, please check {} directory'.format(extract_dir)}
                except BaseException:
                    response = {'retcode': 100,
                                'retmsg': 'download failed, please check if the parameters are correct'}
            else:
                response = response.json()
        return response

    def output_model(self, job_id, role, party_id, component_name):
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        check_config(config=config_data,
                     required_arguments=['job_id', 'component_name', 'role', 'party_id'])
        return self._post(url='tracking/component/output/model', json=config_data)

    def output_data_table(self, job_id, role, party_id, component_name):
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        check_config(config=config_data,
                     required_arguments=['job_id', 'component_name', 'role', 'party_id'])
        return self._post(url='tracking/component/output/data/table', json=config_data)

    def get_summary(self, job_id, role, party_id, component_name):
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        check_config(config=config_data,
                     required_arguments=['job_id', 'component_name', 'role', 'party_id'])
        res = self._post(url='tracking/component/summary/download', handle_result=True, json=config_data)
        if not res.get('data'):
            res['data'] = {}
        return res
