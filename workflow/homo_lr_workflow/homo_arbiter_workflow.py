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

from federatedml.logistic_regression.homo_logsitic_regression import HomoLRArbiter
from federatedml.param import LogisticParam
from federatedml.util import ParamExtract
from federatedml.util import consts
from workflow.homo_lr_workflow.homo_base_workflow import HomoBaseWorkFlow
from arch.api.utils import log_utils

LOGGER = log_utils.getLogger()


class ArbiterWorkFlow(HomoBaseWorkFlow):
    def _initialize_model(self, runtime_conf_path):
        logistic_param = LogisticParam()
        self.logistic_param = ParamExtract.parse_param_from_config(logistic_param, runtime_conf_path)
        self.model = HomoLRArbiter(self.logistic_param)

    def _initialize_role_and_mode(self):
        self.role = consts.ARBITER
        self.mode = consts.HOMO

    def cross_validation(self, data_instance=None):
        for flowid in range(self.workflow_param.n_splits):
            self.model.set_flowid(flowid)
            self.model.fit(data_instance)
            eval_result = self.model.predict(data_instance, self.workflow_param.predict_param)
            LOGGER.debug("Arbiter evaluate result: {}".format(eval_result))
            self._initialize_model(self.config_path)

    def evaluate(self, eval_data):
        LOGGER.info("No need to evaluate")
        pass


if __name__ == "__main__":
    workflow = ArbiterWorkFlow()
    workflow.run()
