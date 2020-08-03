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

from pipeline.component.component_base import Component
from pipeline.backend.config import VERSION


class Input(Component):
    def __init__(self, **kwargs):
        Component.__init__(self, **kwargs)

    def get_config(self, feed_dict):
        """setting args"""
        if not isinstance(feed_dict, dict) or not feed_dict:
            raise ValueError("To run the pipeline, please feed ")

        data_conf = {}

        for role in feed_dict:
            data_conf[role] = {}
            for party_id in feed_dict[role]:
                data = feed_dict[role][party_id]

                data_conf[role][party_id] = {}
                data_conf[role][party_id]["args"] = {self.name: data}

    @property
    def data(self):
        return ".".join(["args", self.name])


