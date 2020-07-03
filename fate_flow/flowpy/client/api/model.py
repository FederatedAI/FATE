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
from arch.api.utils import file_utils
from fate_flow.flowpy.client.api.base import BaseFlowAPI
from fate_flow.flowpy.utils import preprocess


class Model(BaseFlowAPI):
    def load(self, conf_path):
        if not os.path.exists(conf_path):
            raise FileNotFoundError('Invalid conf path, file not exists.')
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        return self._post(url='model/load', json=config_data)

    def bind(self, conf_path):
        if not os.path.exists(conf_path):
            raise FileNotFoundError('Invalid conf path, file not exists.')
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        return self._post(url='model/bind', json=config_data)

    def imp(self, conf_path):
        if not os.path.exists(conf_path):
            raise FileNotFoundError('Invalid conf path, file not exists.')
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        file_path = config_data["file"]
        if not os.path.isabs(file_path):
            file_path = os.path.join(file_utils.get_project_base_directory(), file_path)
        if os.path.exists(file_path):
            files = {'file': open(file_path, 'rb')}
        else:
            raise Exception('The file is obtained from the fate flow client machine, but it does not exist, '
                            'please check the path: {}'.format(file_path))
        return self._post(url='model/import', json=config_data, files=files)

    def export(self, conf_path):
        if not os.path.exists(conf_path):
            raise FileNotFoundError('Invalid conf path, file not exists.')
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
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

    def store(self, conf_path):
        if not os.path.exists(conf_path):
            raise FileNotFoundError('Invalid conf path, file not exists.')
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        return self._post(url='model/store', json=config_data)

    def restore(self, conf_path):
        if not os.path.exists(conf_path):
            raise FileNotFoundError('Invalid conf path, file not exists.')
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        return self._post(url='model/restore', json=config_data)
