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
from federatedml.param import FeatureSelectionParam
from federatedml.util import FeatureSelectionParamChecker
from federatedml.util import ParamExtract
from federatedml.util import consts
from workflow import status_tracer_decorator
from workflow.workflow import WorkFlow

LOGGER = log_utils.getLogger()


class HeteroFeatureSelectGuestWorkflow(WorkFlow):
    def _initialize(self, config_path):
        LOGGER.debug("Get in guest feature selection workflow initialize")

        self._initialize_role_and_mode()
        self._initialize_workflow_param(config_path)
        self._initialize_model(config_path)

    def _initialize_role_and_mode(self):
        self.role = consts.GUEST
        self.mode = consts.HETERO

    def _initialize_model(self, runtime_conf_path):
        feature_param = FeatureSelectionParam()
        self.feature_param = ParamExtract.parse_param_from_config(feature_param, runtime_conf_path)
        FeatureSelectionParamChecker.check_param(self.feature_param)

        # self.model = HeteroFeatureSelectionGuest(self.feature_param)
        # LOGGER.debug("Guest model started")

    @status_tracer_decorator.status_trace
    def run(self):
        self._init_argument()

        filter_methods = self.feature_param.filter_method
        # Step 1: Data io
        if self.feature_param.method in ['fit', 'fit_transform']:
            train_data_instance = self.gen_data_instance(self.workflow_param.train_input_table,
                                                         self.workflow_param.train_input_namespace)
        else:
            train_data_instance = self.gen_data_instance(self.workflow_param.train_input_table,
                                                         self.workflow_param.train_input_namespace,
                                                         mode='transform')
        # Step 2: intersect
        LOGGER.debug("Star intersection before train")
        intersect_flowid = "train_0"
        train_data = self.intersect(train_data_instance, intersect_flowid)
        LOGGER.debug("End intersection before train")

        # Step 3: sample
        sample_flowid = "train_sample_0"
        train_data = self.sample(train_data, sample_flowid)

        # Step 4: binning
        if 'iv_value_thres' in filter_methods or 'iv_percentile' in filter_methods:
            binning_flowid = 'feature_binning'
            train_data = self.feature_binning(data_instances=train_data, flow_id=binning_flowid)

        # Step 5: feature selection
        feature_selection_id = 'feature_selection'
        if self.feature_param.method == 'fit':
            self.feature_selection_fit(data_instance=train_data, flow_id=feature_selection_id, without_transform=True)
        elif self.feature_param.method == 'fit_transform':
            self.feature_selection_fit(data_instance=train_data, flow_id=feature_selection_id)
        else:
            result_table = self.feature_selection_transform(data_instance=train_data, flow_id=feature_selection_id)
            self.save_predict_result(result_table)

        LOGGER.info("Finish guest party feature selection")


if __name__ == "__main__":
    workflow = HeteroFeatureSelectGuestWorkflow()
    workflow.run()
