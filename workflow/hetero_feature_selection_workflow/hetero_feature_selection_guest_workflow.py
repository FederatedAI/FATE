#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
################################################################################
#
#
################################################################################

from arch.api.utils import log_utils
from federatedml.feature.hetero_feature_selection.feature_selection_guest import HeteroFeatureSelectionGuest
from federatedml.param import FeatureSelectionParam
from federatedml.util import FeatureSelectionParamChecker
from federatedml.util import ParamExtract
from federatedml.util import consts
from workflow import status_tracer_decorator
from workflow.workflow import WorkFlow

LOGGER = log_utils.getLogger()


class HeteroFeatureSelectGuestWorkflow(WorkFlow):
    def _initialize(self, config_path):
        self._initialize_role_and_mode()
        self._initialize_model(config_path)
        self._initialize_workflow_param(config_path)

    def _initialize_role_and_mode(self):
        self.role = consts.GUEST
        self.mode = consts.HETERO

    def _initialize_intersect(self, config):
        pass

    def _initialize_model(self, runtime_conf_path):
        feature_param = FeatureSelectionParam()
        self.feature_param = ParamExtract.parse_param_from_config(feature_param, runtime_conf_path)
        FeatureSelectionParamChecker.check_param(self.feature_param)
        self.model = HeteroFeatureSelectionGuest(self.feature_param)
        LOGGER.debug("Guest model started")

    @status_tracer_decorator.status_trace
    def run(self):
        self._init_argument()

        if self.workflow_param.method == "feature_select":

            if self.feature_param.method == 'fit':
                train_data_instance = self.gen_data_instance(self.workflow_param.train_input_table,
                                                             self.workflow_param.train_input_namespace)
                LOGGER.debug("In guest workflow, after data io, schema: {}".format(train_data_instance.schema))
                if self.feature_param.local_only:
                    self.model.fit_local(train_data_instance)
                else:
                    self.model.fit(train_data_instance)
                self.model.save_model(self.workflow_param.model_table, self.workflow_param.model_namespace)

            elif self.feature_param.method == 'fit_transform':
                train_data_instance = self.gen_data_instance(self.workflow_param.train_input_table,
                                                             self.workflow_param.train_input_namespace)
                if self.feature_param.local_only:
                    result_table = self.model.fit_local_transform(train_data_instance)
                else:
                    result_table = self.model.fit_transform(train_data_instance)
                self.model.save_model(self.workflow_param.model_table, self.workflow_param.model_namespace)
                self.save_predict_result(result_table)
                LOGGER.info(
                    "Predict result saved, table: {},"
                    " namespace: {}".format(self.workflow_param.predict_output_table,
                                            self.workflow_param.predict_output_namespace))

            elif self.feature_param.method == 'transform':
                train_data_instance = self.gen_data_instance(self.workflow_param.train_input_table,
                                                             self.workflow_param.train_input_namespace,
                                                             mode='transform')
                LOGGER.debug("In guest workflow, after data io, schema: {}".format(train_data_instance.schema))
                self.load_model()
                result_table = self.model.transform(train_data_instance)
                self.save_predict_result(result_table)
                LOGGER.info(
                    "Predict result saved, table: {},"
                    " namespace: {}".format(self.workflow_param.predict_output_table,
                                            self.workflow_param.predict_output_namespace))

        else:
            raise TypeError("method %s is not support yet" % (self.workflow_param.method))

        LOGGER.info("Finish guest party feature selection")


if __name__ == "__main__":
    workflow = HeteroFeatureSelectGuestWorkflow()
    workflow.run()
