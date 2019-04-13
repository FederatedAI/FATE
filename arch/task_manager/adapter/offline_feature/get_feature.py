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
from arch.api.utils import dtable_utils
from arch.api import storage
from arch.task_manager.settings import logger


class GetFeature(object):
    @staticmethod
    def get_adapter(adapter_name):
        package = os.path.dirname(__file__).replace(os.environ["PYTHONPATH"], "").lstrip("/").replace("/", ".")
        module = "%s.%s" % (package, adapter_name)
        m = importlib.import_module(module)
        return getattr(m, format_transform.underline_to_pascal(adapter_name))

    @staticmethod
    def request(job_id, request_config):
        adapter_name = request_config.get("adapter")
        response = GetFeature.get_adapter(adapter_name=adapter_name).request(job_id)
        logger.info("request offline feature")
        logger.info(response)
        return response

    @staticmethod
    def import_data(job_config):
        adapter_name = job_config.get("adapter")
        input_data = GetFeature.get_adapter(adapter_name=adapter_name).import_data(job_config)
        table_name, table_namespace = dtable_utils.get_table_info(config=job_config, create=True)
        if table_name and table_namespace:
            try:
                storage.save_data(input_data, name=table_name, namespace=table_namespace, partition=50)
                logger.info("import data successfully, table name {}, namespace {}".format(table_name, table_namespace))
                return {"status": 0, "msg": "table name {}, namespace {}".format(table_name, table_namespace)}
            except Exception as e:
                return dict(status=102, msg="save data error: {}".format(str(e)))
        else:
            return dict(status=101, msg="can not get table name and table namespace")
