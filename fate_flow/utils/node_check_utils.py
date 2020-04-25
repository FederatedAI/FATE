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
import functools

import requests
from flask import request

from fate_flow.settings import CHECK_NODES_IDENTITY, MANAGER_HOST, MANAGER_PORT, FATE_MANAGER_NODE_CHECK


def check_nodes(func):
    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        if CHECK_NODES_IDENTITY:
            body = {
                'partyId': request.json.get('src_party_id'),
                'role': request.json.get('src_role'),
                'appKey': request.json.get('appKey'),
                'appSecret': request.json.get('appSecret')
            }
            try:
                response = requests.post(url="http://{}:{}{}".format(MANAGER_HOST, MANAGER_PORT, FATE_MANAGER_NODE_CHECK), json=body).json()
                if response['code'] != 0:
                    raise Exception('Authentication failure: {}'.format(str(response['message'])))
            except Exception as e:
                raise Exception('Authentication error: {}'.format(str(e)))
        return func(*args, **kwargs)
    return _wrapper


def nodes_check(src_party_id, src_role, appKey, appSecret):
    if CHECK_NODES_IDENTITY:
        body = {
            'partyId': src_party_id,
            'role': src_role,
            'appKey': appKey,
            'appSecret': appSecret
        }
        try:
            response = requests.post(url="http://{}:{}{}".format(MANAGER_HOST, MANAGER_PORT, FATE_MANAGER_NODE_CHECK), json=body).json()
            if response['code'] != 0:
                raise Exception(str(response['message']))
        except Exception as e:
            raise Exception('role {} party id {} Authentication error: {}'.format(src_role, src_party_id, str(e)))
