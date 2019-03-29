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
import importlib
from arch.api.utils import format_transform
from arch.api.io import feature
import datetime


class GetFeature(object):
    @staticmethod
    def get_adapter(adapter_name):
        package = os.path.dirname(__file__).replace(os.environ["PYTHONPATH"], "").lstrip("/").replace("/", ".")
        module = "%s.%s" % (package, adapter_name)
        m = importlib.import_module(module)
        return getattr(m, format_transform.underline_to_pascal(adapter_name))

    @staticmethod
    def request(job_id, request_data):
        adapter_name = request_data.get("adapter")
        return GetFeature.get_adapter(adapter_name=adapter_name).request(job_id)

    @staticmethod
    def import_data(request_data, job_config):
        adapter_name = job_config.get("adapter")
        input_data = GetFeature.get_adapter(adapter_name=adapter_name).import_data(request_data)
        commit_id = feature.save_feature_data(input_data,
                                              scene_id=job_config.get("scene_id"),
                                              my_party_id=job_config.get("my_party_id"),
                                              partner_party_id=job_config.get("partner_party_id"),
                                              my_role=job_config.get("partner_party_id"),
                                              commit_log="get feature data at %s" % datetime.datetime.now()
                                              )
        if commit_id:
            return {"status": 0}
        else:
            return {"status": 1}
