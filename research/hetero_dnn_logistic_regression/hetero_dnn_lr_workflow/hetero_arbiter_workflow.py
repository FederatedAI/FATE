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

from research.hetero_dnn_logistic_regression import HeteroDNNLRArbiter
from federatedml.param import LogisticParam
from federatedml.util import consts
from federatedml.util import ParamExtract
from workflow.workflow import WorkFlow


class DNNLRArbiterWorkFlow(WorkFlow):
    def _initialize_model(self, config):
        logistic_param = LogisticParam()
        self.logistic_param = ParamExtract.parse_param_from_config(logistic_param, config)
        self.model = HeteroDNNLRArbiter(self.logistic_param)

    def _initialize_role_and_mode(self):
        self.role = consts.ARBITER
        self.mode = consts.HETERO

    # arbiter do nothing while predict
    def predict(self, data_instance):
        pass


if __name__ == "__main__":
    arbiter_wf = DNNLRArbiterWorkFlow()
    arbiter_wf.run()
