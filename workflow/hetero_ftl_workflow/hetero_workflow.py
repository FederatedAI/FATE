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

from federatedml.param.param import FTLModelParam, FTLLocalModelParam, FTLDataParam
from federatedml.util import ParamExtract
from federatedml.util.transfer_variable import HeteroFTLTransferVariable
from workflow.workflow import WorkFlow
from arch.api.utils import log_utils
LOGGER = log_utils.getLogger()


class FTLWorkFlow(WorkFlow):
    def __init__(self):
        super(FTLWorkFlow, self).__init__()

    def _initialize_model(self, config):
        LOGGER.debug("@ initialize model")
        ftl_model_param = FTLModelParam()
        ftl_local_model_param = FTLLocalModelParam()
        ftl_data_param = FTLDataParam()
        ftl_model_param = ParamExtract.parse_param_from_config(ftl_model_param, config)
        ftl_local_model_param = ParamExtract.parse_param_from_config(ftl_local_model_param, config)
        self.ftl_data_param = ParamExtract.parse_param_from_config(ftl_data_param, config)
        self.ftl_transfer_variable = HeteroFTLTransferVariable()
        self._do_initialize_model(ftl_model_param, ftl_local_model_param, ftl_data_param)

    def _get_transfer_variable(self):
        return self.ftl_transfer_variable

    def _get_data_model_param(self):
        return self.ftl_data_param

    def _do_initialize_model(self, ftl_model_param: FTLModelParam, ftl_local_model_param: FTLLocalModelParam,
                             ftl_data_param: FTLDataParam):
        raise NotImplementedError("method init must be define")

    def run(self):
        self._init_argument()
        if self.workflow_param.method == "train":
            data_instance = self.gen_data_instance(self.workflow_param.predict_input_table,
                                                   self.workflow_param.predict_input_namespace)
            self.train(data_instance)

        elif self.workflow_param.method == "predict":
            data_instance = self.gen_data_instance(self.workflow_param.predict_input_table,
                                                   self.workflow_param.predict_input_namespace)
            self.predict(data_instance)
        else:
            raise TypeError("method %s is not support yet" % (self.workflow_param.method))
