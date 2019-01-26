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

import json
import uuid
import sys
from workflow.hetero_ftl_workflow.hetero_guest_workflow import FTLGuestWorkFlow
from federatedml.ftl.common.data_util import load_guest_host_dtable_from_UCI_Credit_Card, generate_table_namespace_n_name
from arch.api import eggroll
from arch.api import federation
from arch.api.utils import log_utils
LOGGER = log_utils.getLogger()

config_path = "./conf/guest_runtime_conf.json"


class TestFTLGuest(FTLGuestWorkFlow):

    def __init__(self):
        super(TestFTLGuest, self).__init__()

    def _init_argument(self):
        self._initialize(config_path)
        with open(config_path) as conf_f:
            runtime_json = json.load(conf_f)

        LOGGER.debug("The Guest job id is {}".format(job_id))
        LOGGER.debug("The Guest work mode id is {}".format(self.workflow_param.work_mode))
        eggroll.init(job_id, self.workflow_param.work_mode)
        federation.init(job_id, runtime_json)
        LOGGER.debug("Finish eggroll and federation init")

    def gen_data_instance(self, table_name, namespace):
        data_model = self._get_data_model_param()
        if data_model.is_read_table:
            return eggroll.table(table_name, namespace)
        else:

            file_path = data_model.file_path
            overlap_ratio = data_model.overlap_ratio
            guest_split_ratio = data_model.guest_split_ratio
            guest_feature_num = data_model.n_feature_guest
            num_samples = data_model.num_samples
            balanced = data_model.balanced

            namespace, table_name = generate_table_namespace_n_name(file_path)
            suffix = "_" + str(uuid.uuid1())
            tables_name = {
                "guest_table_ns": "guest_" + namespace + suffix,
                "guest_table_name": "guest_" + table_name + suffix,
                "host_table_ns": "host_" + namespace + suffix,
                "host_table_name": "host_" + table_name + suffix,
            }

            guest_data, host_data = load_guest_host_dtable_from_UCI_Credit_Card(file_path=file_path,
                                                                                num_samples=num_samples,
                                                                                tables_name=tables_name,
                                                                                overlap_ratio=overlap_ratio,
                                                                                guest_split_ratio=guest_split_ratio,
                                                                                guest_feature_num=guest_feature_num,
                                                                                balanced=balanced)
            return guest_data


if __name__ == '__main__':
    job_id = sys.argv[1]
    workflow = TestFTLGuest()
    workflow.run()
