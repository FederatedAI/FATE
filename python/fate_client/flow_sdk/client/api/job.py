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
import json
from contextlib import closing
from flow_sdk.client.api.base import BaseFlowAPI
from flow_sdk.utils import preprocess, check_config, download_from_request, get_parser_version_set


class Job(BaseFlowAPI):
    def list(self, limit=10):
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        return self._post(url='job/list/job', json=config_data)

    def view(self, job_id=None, role=None, party_id=None, status=None):
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        return self._post(url='job/data/view/query', json=config_data)

    def submit(self, conf_path, dsl_path=None):
        if not os.path.exists(conf_path):
            raise FileNotFoundError('Invalid conf path, file not exists.')

        kwargs = locals()

        config_data, dsl_data = preprocess(**kwargs)
        post_data = {
            'job_dsl': dsl_data,
            'job_runtime_conf': config_data
        }

        return self._post(url='job/submit', json=post_data)

    def stop(self, job_id):
        job_id = str(job_id)
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        check_config(config=config_data, required_arguments=['job_id'])
        return self._post(url='job/stop', json=config_data)

    def query(self, job_id=None, role=None, party_id=None, component_name=None, status=None):
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        return self._post(url='job/query', json=config_data)

    def config(self, job_id, role, party_id, output_path):
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        check_config(config=config_data, required_arguments=['job_id', 'role', 'party_id', 'output_path'])
        response = self._post(url='job/config', json=config_data)

        if response['retcode'] == 0:
            job_id = response['data']['job_id']
            download_directory = os.path.join(config_data['output_path'], 'job_{}_config'.format(job_id))
            os.makedirs(download_directory, exist_ok=True)
            for k, v in response['data'].items():
                if k == 'job_id':
                    continue
                with open('{}/{}.json'.format(download_directory, k), 'w') as fw:
                    json.dump(v, fw, indent=4)
            del response['data']['dsl']
            del response['data']['runtime_conf']
            response['directory'] = download_directory
            response['retmsg'] = 'download successfully, please check {} directory'.format(download_directory)

        return response

    def log(self, job_id, output_path):
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        check_config(config=config_data, required_arguments=['job_id', 'output_path'])
        job_id = config_data['job_id']
        tar_file_name = 'job_{}_log.tar.gz'.format(job_id)
        extract_dir = os.path.join(config_data['output_path'], 'job_{}_log'.format(job_id))
        with closing(self._get(url='job/log', handle_result=False, json=config_data, stream=True)) as response:
            if response.status_code == 200:
                download_from_request(http_response=response, tar_file_name=tar_file_name, extract_dir=extract_dir)
                response = {'retcode': 0,
                            'directory': extract_dir,
                            'retmsg': 'download successfully, please check {} directory'.format(extract_dir)}
            else:
                response = response.json()
        return response

    def generate_dsl(self, train_dsl_path, cpn_file_path=None, cpn_list: list = None):
        kwargs = locals()
        if not os.path.exists(kwargs.get("train_dsl_path")):
            raise Exception("Train dsl file not exists.")
        if kwargs.get("cpn_list"):
            cpn_str = kwargs.get("cpn_list")
        elif kwargs.get("cpn_file_path"):
            with open(kwargs.get("cpn_file_path"), "r") as fp:
                cpn_str = fp.read()
        else:
            cpn_str = ""

        with open(kwargs.get("train_dsl_path"), "r") as ft:
            train_dsl = ft.read()

        config_data = {
            "cpn_str": cpn_str,
            "train_dsl": train_dsl,
            "version": "2"
        }

        res = self._post(url="job/dsl/generate", handle_result=True, json=config_data)
        if not res.get("data"):
            res["data"] = {}
        return res

    # TODO complete it in next version
    # def clean(self, job_id=None, role=None, party_id=None, component_name=None):
    #     kwargs = locals()
    #     config_data, dsl_data = preprocess(**kwargs)
    #     check_config(config=config_data, required_arguments=['job_id'])
    #     return self._post(url='job/clean', json=config_data)
