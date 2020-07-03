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


class BaseFlowAPI:

    def __init__(self, client=None):
        self._client = client

    def _get(self, url, handle_result=True, **kwargs):
        if handle_result:
            return self._handle_result(self._client.get(url, **kwargs))
        else:
            return self._client.get(url, **kwargs)

    def _post(self, url, handle_result=True, **kwargs):
        if handle_result:
            return self._handle_result(self._client.post(url, **kwargs))
        else:
            return self._client.post(url, **kwargs)

    def _handle_result(self, response):
        return self._client._handle_result(response)

    @property
    def session(self):
        return self._client.session

    @property
    def ip(self):
        return self._client.ip

    @property
    def port(self):
        return self._client.port

    @property
    def version(self):
        return self._client.version
