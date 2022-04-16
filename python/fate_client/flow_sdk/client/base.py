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
import inspect
import json
import sys
import traceback
from base64 import b64encode
from hmac import HMAC
from time import time
from urllib.parse import quote, urlencode
from uuid import uuid1

import requests

from flow_sdk.client.api.base import BaseFlowAPI


def _is_api_endpoint(obj):
    return isinstance(obj, BaseFlowAPI)


class BaseFlowClient:
    API_BASE_URL = ''

    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls)
        api_endpoints = inspect.getmembers(self, _is_api_endpoint)
        for name, api in api_endpoints:
            api_cls = type(api)
            api = api_cls(self)
            setattr(self, name, api)
        return self

    def __init__(self, ip, port, version, app_key=None, secret_key=None):
        self._http = requests.Session()
        self.ip = ip
        self.port = port
        self.version = version

        self.app_key = app_key if app_key and secret_key else None
        self.secret_key = secret_key if app_key and secret_key else None

    def _request(self, method, url, **kwargs):
        stream = kwargs.pop('stream', self._http.stream)
        prepped = requests.Request(method, self.API_BASE_URL + url, **kwargs).prepare()

        if self.app_key and self.secret_key:
            timestamp = str(round(time() * 1000))
            nonce = str(uuid1())
            signature = b64encode(HMAC(self.secret_key.encode('ascii'), b'\n'.join([
                timestamp.encode('ascii'),
                nonce.encode('ascii'),
                self.app_key.encode('ascii'),
                prepped.path_url.encode('ascii'),
                prepped.body if kwargs.get('json') else b'',
                urlencode(sorted(kwargs['data'].items()), quote_via=quote, safe='-._~').encode('ascii')
                if kwargs.get('data') and isinstance(kwargs['data'], dict) else b'',
            ]), 'sha1').digest()).decode('ascii')

            prepped.headers.update({
                'TIMESTAMP': timestamp,
                'NONCE': nonce,
                'APP_KEY': self.app_key,
                'SIGNATURE': signature,
            })

        try:
            response = self._http.send(prepped, stream=stream)
        except Exception as e:
            response = {
                'retcode': 100,
                'retmsg': str(e),
            }

            if 'connection refused' in response['retmsg'].lower():
                response['retmsg'] = 'Connection refused, Please check if the fate flow service is started'
            else:
                exc_type, exc_value, exc_traceback_obj = sys.exc_info()
                response['traceback'] = traceback.format_exception(exc_type, exc_value, exc_traceback_obj)

        return response

    @staticmethod
    def _decode_result(response):
        try:
            result = json.loads(response.content.decode('utf-8', 'ignore'), strict=False)
        except (TypeError, ValueError):
            return response
        else:
            return result

    def _handle_result(self, response):
        try:
            if isinstance(response, requests.models.Response):
                return response.json()
            elif isinstance(response, dict):
                return response
            else:
                return self._decode_result(response)
        except json.decoder.JSONDecodeError:
            res = {'retcode': 100,
                   'retmsg': "Internal server error. Nothing in response. You may check out the configuration in "
                   "'FATE/conf/service_conf.yaml' and restart fate flow server."}
            return res

    def get(self, url, **kwargs):
        return self._request(method='get', url=url, **kwargs)

    def post(self, url, **kwargs):
        return self._request(method='post', url=url, **kwargs)
