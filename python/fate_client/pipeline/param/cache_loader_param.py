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


class CacheLoaderParam:
    def __init__(self, cache_key=None, job_id=None, component_name=None, cache_name=None):
        super().__init__()
        self.cache_key = cache_key
        self.job_id = job_id
        self.component_name = component_name
        self.cache_name = cache_name

    def check(self):
        return True
