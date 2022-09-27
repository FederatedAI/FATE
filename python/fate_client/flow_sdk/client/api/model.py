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
import re
from contextlib import closing
from flow_sdk.client.api.base import BaseFlowAPI
from flow_sdk.utils import preprocess, get_project_base_directory


class Model(BaseFlowAPI):
    def load(self, config_data=None, job_id=None):
        if config_data is None and job_id is None:
            return {
                "retcode": 100,
                "retmsg": "Load model failed. No arguments received, "
                          "please provide one of arguments from job id and conf path."
            }
        if config_data is not None and job_id is not None:
            return {
                "retcode": 100,
                "retmsg": "Load model failed. Please do not provide job id and "
                          "conf path at the same time."
            }

        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        return self._post(url='model/load', json=config_data)

    def bind(self, config_data, job_id=None):
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        return self._post(url='model/bind', json=config_data)

    def import_model(self, config_data, from_database=False):
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)

        if kwargs.pop('from_database'):
            return self._post(url='model/restore', json=config_data)

        file_path = config_data['file']

        if not os.path.isabs(file_path):
            file_path = os.path.join(get_project_base_directory(), file_path)

        if os.path.exists(file_path):
            FileNotFoundError(
                'The file is obtained from the fate flow client machine, but it does not exist, '
                ' please check the path: {}'.format(file_path)
            )

        config_data['force_update'] = int(config_data.get('force_update', False))
        files = {'file': open(file_path, 'rb')}
        return self._post(url='model/import', data=config_data, files=files)

    def export_model(self, config_data, to_database=False):
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        if not config_data.pop("to_database"):
            with closing(self._get(url='model/export', handle_result=False, json=config_data, stream=True)) as response:
                if response.status_code == 200:
                    archive_file_name = re.findall("filename=(.+)", response.headers["Content-Disposition"])[0]
                    os.makedirs(config_data["output_path"], exist_ok=True)
                    archive_file_path = os.path.join(config_data["output_path"], archive_file_name)
                    with open(archive_file_path, 'wb') as fw:
                        for chunk in response.iter_content(1024):
                            if chunk:
                                fw.write(chunk)
                    response = {'retcode': 0,
                                'file': archive_file_path,
                                'retmsg': 'download successfully, please check {}'.format(archive_file_path)}
                else:
                    response = response.json()
            return response
        return self._post(url='model/store', json=config_data)

    def migrate(self, config_data):
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        return self._post(url='model/migrate', json=config_data)

    def tag_model(self, job_id, tag_name, remove=False):
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        if not config_data.pop('remove'):
            return self._post(url='model/model_tag/create', json=config_data)
        else:
            return self._post(url='model/model_tag/remove', json=config_data)

    def tag_list(self, job_id):
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        return self._post(url='model/model_tag/retrieve', json=config_data)

    def deploy(self, model_id, model_version, cpn_list=None, predict_dsl=None, components_checkpoint=None):
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        return self._post(url='model/deploy', json=config_data)

    def get_predict_dsl(self, model_id, model_version):
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        return self._post(url='model/get/predict/dsl', json=config_data)

    def get_predict_conf(self, model_id, model_version):
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        return self._post(url='model/get/predict/conf', json=config_data)

    def get_model_info(self, model_id=None, model_version=None, role=None, party_id=None, query_filters=None, **kwargs):
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        return self._post(url='model/query', json=config_data)

    def homo_convert(self, config_data):
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        return self._post(url='model/homo/convert', json=config_data)

    def homo_deploy(self, config_data):
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        if config_data.get('deployment_type') == "kfserving":
            kube_config = config_data.get('deployment_parameters', {}).get('config_file')
            if kube_config:
                if not os.path.isabs(kube_config):
                    kube_config = os.path.join(get_project_base_directory(), kube_config)
                if os.path.exists(kube_config):
                    with open(kube_config, 'r') as fp:
                        config_data['deployment_parameters']['config_file_content'] = fp.read()
                    del config_data['deployment_parameters']['config_file']
                else:
                    raise Exception('The kube_config file is obtained from the fate flow client machine, '
                                    'but it does not exist, please check the path: {}'.format(kube_config))
        return self._post(url='model/homo/deploy', json=config_data)
