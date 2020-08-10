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

from arch.api.utils import file_utils


class FederationAuthorization(object):

    def __init__(self, transfer_conf_path):
        self.transfer_auth = {}
        for path, _, file_names in os.walk(os.path.join(file_utils.get_project_base_directory(), transfer_conf_path)):
            for name in file_names:
                transfer_conf = os.path.join(path, name)
                if transfer_conf.endswith(".json"):
                    self.transfer_auth.update(file_utils.load_json_conf(transfer_conf))

        # cache
        self._authorized_src = {}
        self._authorized_dst = {}

    def _update_auth(self, variable_name):
        a_name, v_name = variable_name.split(".", 1)
        variable_auth = self.transfer_auth.get(a_name, {}).get(v_name, None)
        if variable_auth is None:
            raise ValueError(f"Unauthorized variable: {v_name}")
        auth_src = variable_auth["src"]
        if not isinstance(auth_src, list):
            auth_src = [auth_src]
        auth_dst = variable_auth["dst"]
        self._authorized_src[variable_name] = auth_src
        self._authorized_dst[variable_name] = auth_dst

    def authorized_src_roles(self, variable_name):
        if variable_name not in self._authorized_src:
            self._update_auth(variable_name)
        return self._authorized_src[variable_name]

    def authorized_dst_roles(self, variable_name):
        if variable_name not in self._authorized_dst:
            self._update_auth(variable_name)
        return self._authorized_dst[variable_name]
