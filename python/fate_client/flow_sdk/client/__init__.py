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
from flow_sdk.client.base import BaseFlowClient
from flow_sdk.client import api


class FlowClient(BaseFlowClient):
    job = api.Job()
    component = api.Component()
    data = api.Data()
    queue = api.Queue()
    table = api.Table()
    task = api.Task()
    model = api.Model()
    tag = api.Tag()
    privilege = api.Privilege()
    checkpoint = api.Checkpoint()
    remote_version = api.Version()
    test = api.Test()

    def __init__(self, ip, port, version, app_key=None, secret_key=None):
        super().__init__(ip, port, version, app_key, secret_key)
        self.API_BASE_URL = 'http://%s:%s/%s/' % (ip, port, version)
