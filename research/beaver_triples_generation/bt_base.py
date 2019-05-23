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

from arch.api import federation

from arch.api.utils import log_utils

LOGGER = log_utils.getLogger()


class BaseBeaverTripleGeneration(object):

    def _do_remote(self, value=None, name=None, tag=None, role=None, idx=None):
        federation.remote(value, name=name, tag=tag, role=role, idx=idx)

    def _do_get(self, name=None, tag=None, idx=None):
        return federation.get(name=name, tag=tag, idx=idx)

    def save_beaver_triples(self, bt_map, bt_map_name):
        LOGGER.debug("@ save bt map with name:" + bt_map_name)
        # TODO: Save beaver triples persistently

