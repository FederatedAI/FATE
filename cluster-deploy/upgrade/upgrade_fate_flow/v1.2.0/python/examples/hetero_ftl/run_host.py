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
import sys

from workflow.hetero_ftl_workflow.hetero_host_workflow import FTLHostWorkFlow
from federatedml.ftl.data_util.uci_credit_card_util import load_guest_host_dtable_from_UCI_Credit_Card
from arch.api import session
from arch.api import federation
from arch.api.utils import log_utils
LOGGER = log_utils.getLogger()

config_path = "./conf/host_runtime_conf.json"


class TestFTLHost(FTLHostWorkFlow):

    def __init__(self):
        super(TestFTLHost, self).__init__()

    def _init_argument(self):
        with open(config_path) as conf_f:
            runtime_json = json.load(conf_f)
        self._initialize(runtime_json)

        LOGGER.debug("The Host job id is {}".format(job_id))
        LOGGER.debug("The Host work mode id is {}".format(self.workflow_param.work_mode))
        session.init(job_id, self.workflow_param.work_mode)
        federation.init(job_id, runtime_json)
        LOGGER.debug("Finish eggroll and federation init")

    def gen_data_instance(self, table_name, namespace, mode="fit"):
        data_model_param = self._get_data_model_param()
        if data_model_param.is_read_table:
            return session.table(table_name, namespace)
        else:
            data_model_param_dict = dict()
            data_model_param_dict["file_path"] = data_model_param.file_path
            data_model_param_dict["num_samples"] = data_model_param.num_samples
            data_model_param_dict["overlap_ratio"] = data_model_param.overlap_ratio
            data_model_param_dict["guest_split_ratio"] = data_model_param.guest_split_ratio
            data_model_param_dict["n_feature_guest"] = data_model_param.n_feature_guest
            data_model_param_dict["balanced"] = data_model_param.balanced
            _, host_data = load_guest_host_dtable_from_UCI_Credit_Card(data_model_param_dict)
            return host_data

    def gen_validation_data_instance(self, table_name, namespace):
        data_model_param = self._get_data_model_param()
        valid_data_model_param = self._get_valid_data_model_param()
        if valid_data_model_param.is_read_table:
            return session.table(table_name, namespace)
        else:
            data_model_param_dict = dict()
            data_model_param_dict["file_path"] = valid_data_model_param.file_path
            data_model_param_dict["num_samples"] = valid_data_model_param.num_samples
            data_model_param_dict["overlap_ratio"] = data_model_param.overlap_ratio
            data_model_param_dict["guest_split_ratio"] = data_model_param.guest_split_ratio
            data_model_param_dict["n_feature_guest"] = data_model_param.n_feature_guest
            data_model_param_dict["balanced"] = data_model_param.balanced
            _, host_data = load_guest_host_dtable_from_UCI_Credit_Card(data_model_param_dict)
            return host_data


if __name__ == '__main__':
    job_id = sys.argv[1]
    workflow = TestFTLHost()
    workflow.run()
