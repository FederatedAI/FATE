#
#  Copyright 2021 The FATE Authors. All Rights Reserved.
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
from flow_sdk.client.api.base import BaseFlowAPI


class Version(BaseFlowAPI):

    def api(self):
        return self._post(url='version/get').get('data', {}).get('API')

    def fate(self):
        return self._post(url='version/get', json={'module': 'FATE'}).get('data', {}).get('FATE')

    def fate_flow(self):
        return self._post(url='version/get', json={'module': 'FATEFlow'}).get('data', {}).get('FATEFlow')

    def fate_board(self):
        return self._post(url='version/get', json={'module': 'FATEBoard'}).get('data', {}).get('FATEBoard')

    def centos(self):
        return self._post(url='version/get', json={'module': 'CENTOS'}).get('data', {}).get('CENTOS')

    def ubuntu(self):
        return self._post(url='version/get', json={'module': 'UBUNTU'}).get('data', {}).get('UBUNTU')

    def python(self):
        return self._post(url='version/get', json={'module': 'PYTHON'}).get('data', {}).get('PYTHON')

    def jdk(self):
        return self._post(url='version/get', json={'module': 'JDK'}).get('data', {}).get('JDK')

    def maven(self):
        return self._post(url='version/get', json={'module': 'MAVEN'}).get('data', {}).get('MAVEN')

    def eggroll(self):
        return self._post(url='version/get', json={'module': 'EGGROLL'}).get('data', {}).get('EGGROLL')

    def spark(self):
        return self._post(url='version/get', json={'module': 'SPARK'}).get('data', {}).get('SPARK')
