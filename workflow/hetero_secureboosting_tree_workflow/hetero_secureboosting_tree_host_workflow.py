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

# =============================================================================
# Base WorkFlow Define
# =============================================================================

from federatedml.param import BoostingTreeParam
from federatedml.util import ParamExtract
from federatedml.util import consts
from federatedml.tree import HeteroSecureBoostingTreeHost
from workflow.workflow import WorkFlow
import json
import sys


class HeteroSecureBoostingTreeHostWorkFlow(WorkFlow):

    def _initialize_model(self, config):

        secureboosting_param = BoostingTreeParam()
        self.secureboosting_tree_param = ParamExtract.parse_param_from_config(secureboosting_param, config)
        self.model = HeteroSecureBoostingTreeHost(self.secureboosting_tree_param)
        self._set_runtime_idx(config)

    def _initialize_role_and_mode(self):
        self.role = consts.HOST
        self.mode = consts.HETERO

    def _set_runtime_idx(self, config_path):
        with open(config_path) as conf_f:
            runtime_json = json.load(conf_f)
        """get runtime index"""
        local_role = runtime_json["local"]["role"]
        local_partyid = runtime_json["local"]["party_id"]
        runtime_idx = runtime_json["role"][local_role].index(local_partyid)
        self.model.set_runtime_idx(runtime_idx)

    def save_predict_result(self, predict_result):
        pass

    def evaluate(self, eval_data):  
        pass

    def save_eval_result(self, eval_data):
        pass


if __name__ == "__main__":
    workflow = HeteroSecureBoostingTreeHostWorkFlow()
    workflow.run()


