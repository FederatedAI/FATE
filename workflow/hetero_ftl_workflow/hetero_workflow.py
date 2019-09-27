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

import numpy as np

from arch.api import session
from arch.api.utils import log_utils
from federatedml.param.ftl_param import FTLModelParam, LocalModelParam, FTLDataParam, FTLValidDataParam
from federatedml.param.workflow_param import WorkFlowParam
from federatedml.util.param_extract import ParamExtract
from federatedml.transfer_variable.transfer_class.hetero_ftl_transfer_variable import HeteroFTLTransferVariable

LOGGER = log_utils.getLogger()


class FTLWorkFlow(object):
    def __init__(self):
        super(FTLWorkFlow, self).__init__()
        self.model = None
        self.job_id = None
        self.workflow_param = None
        self.param_extract = None

    def _initialize(self, config):
        LOGGER.debug("Get in base workflow initialize")
        self._initialize_model(config)
        self._initialize_workflow_param(config)

    def _initialize_model(self, config):
        LOGGER.debug("@ initialize model")
        ftl_model_param = FTLModelParam()
        ftl_local_model_param = LocalModelParam()
        ftl_data_param = FTLDataParam()
        ftl_valid_data_param = FTLValidDataParam()

        self.param_extract = ParamExtract()
        ftl_model_param = self.param_extract.parse_param_from_config(ftl_model_param, config)
        ftl_local_model_param = self.param_extract.parse_param_from_config(ftl_local_model_param, config)
        self.ftl_data_param = self.param_extract.parse_param_from_config(ftl_data_param, config)
        self.ftl_valid_data_param = self.param_extract.parse_param_from_config(ftl_valid_data_param, config)
        self.ftl_transfer_variable = HeteroFTLTransferVariable()

        FTLModelParam.check(ftl_model_param)
        LocalModelParam.check(ftl_local_model_param)
        FTLDataParam.check(self.ftl_data_param)
        FTLValidDataParam.check(self.ftl_valid_data_param)

        self._do_initialize_model(ftl_model_param, ftl_local_model_param, self.ftl_data_param)

    def _initialize_workflow_param(self, config):
        workflow_param = WorkFlowParam()
        self.workflow_param = self.param_extract.parse_param_from_config(workflow_param, config)
        workflow_param.check()

    def _get_transfer_variable(self):
        return self.ftl_transfer_variable

    def _get_data_model_param(self):
        return self.ftl_data_param

    def _get_valid_data_model_param(self):
        return self.ftl_valid_data_param

    def _do_initialize_model(self, ftl_model_param: FTLModelParam, ftl_local_model_param: LocalModelParam,
                             ftl_data_param: FTLDataParam):
        raise NotImplementedError("method init must be define")

    def save_eval_result(self, eval_data):
        LOGGER.info("@ save evaluation result to table with namespace: {0} and name: {1}".format(
            self.workflow_param.evaluation_output_namespace, self.workflow_param.evaluation_output_table))
        session.parallelize([eval_data],
                            include_key=False,
                            name=self.workflow_param.evaluation_output_table,
                            namespace=self.workflow_param.evaluation_output_namespace,
                            error_if_exist=False,
                            persistent=True
                            )

    def save_predict_result(self, predict_result):
        LOGGER.info("@ save prediction result to table with namespace: {0} and name: {1}".format(
            self.workflow_param.predict_output_namespace, self.workflow_param.predict_output_table))
        predict_result.save_as(self.workflow_param.predict_output_table, self.workflow_param.predict_output_namespace)

    def evaluate(self, eval_data):
        if eval_data is None:
            LOGGER.info("not eval_data!")
            return None

        eval_data_local = eval_data.collect()
        labels = []
        pred_prob = []
        pred_labels = []
        data_num = 0
        for data in eval_data_local:
            data_num += 1
            labels.append(data[1][0])
            pred_prob.append(data[1][1])
            pred_labels.append(data[1][2])

        labels = np.array(labels)
        pred_prob = np.array(pred_prob)
        pred_labels = np.array(pred_labels)

        evaluation_result = self.model.evaluate(labels, pred_prob, pred_labels,
                                                evaluate_param=self.workflow_param.evaluate_param)
        return evaluation_result

    def _init_argument(self):
        pass

    def gen_validation_data_instance(self, table, namespace):
        pass

    def gen_data_instance(self, table, namespace):
        pass

    def train(self, train_data_instance, validation_data=None):
        pass

    def predict(self, data_instance):
        pass

    def run(self):
        self._init_argument()
        if self.workflow_param.method == "train":
            data_instance = self.gen_data_instance(self.workflow_param.train_input_table,
                                                   self.workflow_param.train_input_namespace)

            valid_instance = self.gen_validation_data_instance(self.workflow_param.predict_input_table,
                                                               self.workflow_param.predict_input_namespace)
            self.train(data_instance, valid_instance)

        elif self.workflow_param.method == "predict":
            data_instance = self.gen_data_instance(self.workflow_param.predict_input_table,
                                                   self.workflow_param.predict_input_namespace)
            self.predict(data_instance)
        else:
            raise TypeError("method %s is not support yet" % (self.workflow_param.method))
