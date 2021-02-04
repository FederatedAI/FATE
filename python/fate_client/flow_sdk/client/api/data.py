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
import sys
from flow_sdk.client.api.base import BaseFlowAPI
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
from flow_sdk.utils import preprocess, start_cluster_standalone_job_server, get_project_base_directory, json_dumps


class Data(BaseFlowAPI):
    def upload(self, conf_path, verbose=0, drop=0):
        kwargs = locals()
        kwargs['drop'] = int(kwargs['drop']) if int(kwargs['drop']) else 2
        kwargs['verbose'] = int(kwargs['verbose'])
        config_data, dsl_data = preprocess(**kwargs)
        if config_data.get('use_local_data', 1):
            file_name = config_data.get('file')
            if not os.path.isabs(file_name):
                file_name = os.path.join(get_project_base_directory(), file_name)
            if os.path.exists(file_name):
                with open(file_name, 'rb') as fp:
                    data = MultipartEncoder(
                        fields={'file': (os.path.basename(file_name), fp, 'application/octet-stream')}
                    )
                    tag = [0]

                    def read_callback(monitor):
                        if config_data.get('verbose') == 1:
                            sys.stdout.write(
                                "\r UPLOADING:{0}{1}".format("|" * (monitor.bytes_read * 100 // monitor.len),
                                                             '%.2f%%' % (monitor.bytes_read * 100 // monitor.len)))
                            sys.stdout.flush()
                            if monitor.bytes_read / monitor.len == 1:
                                tag[0] += 1
                                if tag[0] == 2:
                                    sys.stdout.write('\n')

                    data = MultipartEncoderMonitor(data, read_callback)
                    return self._post(url='data/upload', data=data,
                                      params=json_dumps(config_data), headers={'Content-Type': data.content_type})
            else:
                raise Exception('The file is obtained from the fate flow client machine, but it does not exist, '
                                'please check the path: {}'.format(file_name))
        else:
            return self._post(url='data/upload', json=config_data)

    def download(self, conf_path):
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        response = self._post(url='data/download', json=config_data)
        try:
            if response['retcode'] == 999:
                start_cluster_standalone_job_server()
                return self._post(url='data/download', json=config_data)
            else:
                return response
        except:
            pass

    def upload_history(self, limit=10, job_id=None):
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        response = self._post(url='data/upload/history', json=config_data)
        try:
            if response['retcode'] == 999:
                start_cluster_standalone_job_server()
                return self._post(url='data/upload/history', json=config_data)
            else:
                return response
        except:
            pass

    def download_history(self):
        pass
