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
class JobConf(object):
    def __init__(self):
        self._conf = dict()

    def set(self, k, v):
        self._conf[k] = v

    def set_all(self, **kwargs):
        self._conf.update(kwargs)

    def update(self, conf: dict):
        for k, v in conf.items():
            if k not in self._conf:
                self._conf[k] = v

    def dict(self):
        return self._conf


class TaskConf(JobConf):
    ...
