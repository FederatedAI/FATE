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
import requests

from fate_flow.settings import CHECK_NODES_IDENTITY, FATE_MANAGER_NODE_CHECK_ENDPOINT
from fate_flow.db.service_registry import ServiceRegistry


def nodes_check(src_party_id, src_role, appKey, appSecret, dst_party_id):
    if CHECK_NODES_IDENTITY:
        body = {
            'srcPartyId': int(src_party_id),
            'role': src_role,
            'appKey': appKey,
            'appSecret': appSecret,
            'dstPartyId': int(dst_party_id),
            'federatedId': ServiceRegistry.FATEMANAGER.get("federatedId")
        }
        try:
            response = requests.post(url="http://{}:{}{}".format(
                ServiceRegistry.FATEMANAGER.get("host"),
                ServiceRegistry.FATEMANAGER.get("port"),
                FATE_MANAGER_NODE_CHECK_ENDPOINT), json=body).json()
            if response['code'] != 0:
                raise Exception(str(response['msg']))
        except Exception as e:
            raise Exception('role {} party id {} authentication failed: {}'.format(src_role, src_party_id, str(e)))
