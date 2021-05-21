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
    def load(self, conf_path=None, job_id=None):
        kwargs = locals()
        if not kwargs.get("conf_path") and not kwargs.get("job_id"):
            response = {
                "retcode": 100,
                "retmsg": "Load model failed. No arguments received, "
                          "please provide one of arguments from job id and conf path."
            }
        else:
            if kwargs.get("conf_path") and kwargs.get("job_id"):
                response = {
                    "retcode": 100,
                    "retmsg": "Load model failed. Please do not provide job id and "
                              "conf path at the same time."
                }
            else:
                config_data, dsl_data = preprocess(**kwargs)
                self._post(url='model/load', json=config_data)
        if not os.path.exists(conf_path):
            raise FileNotFoundError('Invalid conf path, file not exists.')
        config_data, dsl_data = preprocess(**kwargs)
        return self._post(url='model/load', json=config_data)

    def bind(self, conf_path, job_id=None):
        if not os.path.exists(conf_path):
            raise FileNotFoundError('Invalid conf path, file not exists.')
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        return self._post(url='model/bind', json=config_data)

    def import_model(self, conf_path, from_database=False):
        if not os.path.exists(conf_path):
            raise FileNotFoundError('Invalid conf path, file not exists.')
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        if not kwargs.pop("from_database"):
            file_path = config_data["file"]
            if not os.path.isabs(file_path):
                file_path = os.path.join(get_project_base_directory(), file_path)
            if os.path.exists(file_path):
                files = {'file': open(file_path, 'rb')}
            else:
                raise Exception('The file is obtained from the fate flow client machine, but it does not exist, '
                                'please check the path: {}'.format(file_path))
            return self._post(url='model/import', data=config_data, files=files)
        return self._post(url='model/restore', json=config_data)

    def export_model(self, conf_path, to_database=False):
        if not os.path.exists(conf_path):
            raise FileNotFoundError('Invalid conf path, file not exists.')
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

    def migrate(self, conf_path):
        if not os.path.exists(conf_path):
            raise FileNotFoundError('Invalid conf path, file not exists.')
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

    def deploy(self, model_id, model_version, cpn_list=None, predict_dsl=None):
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

