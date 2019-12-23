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

from arch.api import session   
from arch.api.utils import log_utils
from federatedml.evaluation.evaluation import IC
from federatedml.statistic import data_overview
from federatedml.model_selection.stepwise.step import Step
from federatedml.transfer_variable.transfer_class.stepwise_transfer_variable import StepwiseTransferVariable
from federatedml.util import consts

import itertools

import numpy as np

LOGGER = log_utils.getLogger()
session.init("stepwise")

class ModelInfo(object):
    def __init__(self, step_direction, n_step, n_model, score):
        self.score = score
        self.n_step = n_step
        self.n_model = n_model
        self.step_direction = step_direction
        
    def get_key(self):
        """
        get key to obtain model info from models table
        """
        return "{}.{}.{}".format(self.step_direction, self.n_step, self.n_model)

    def get_score(self):
        return self.score 

class HeteroStepwise(object):
    def __init__(self):
        self.mode = None
        self.role = None
        self.forward = False
        self.backward = False
        self.step_direction = ""
        self.best_list = []
        self.n_step = 0
        self.has_test = False
        self.n_count = 0

    def _init_model(self, param):
        self.model_param = param
        self.mode = param.mode
        self.role = param.role
        self.score_name = param.score_name
        self.direction = param.direction
        self.p_enter = param.p_enter
        self.p_remove = param.p_remove
        self.max_step = param.max_step
        self.score = param.score
        self.transfer_variable = StepwiseTransferVariable()
        self._get_direction()
        self._make_table()
        # only used by Arbiter to fast filter model
        self.models_trained = {}

    def _get_direction(self):
        if self.direction == "forward":
            self.forward = True
        elif self.direction == "backward":
            self.backward = True
        elif self.direction == "both":
            self.forward = True
            self.backward = True
        else:
            LOGGER.warning("Wrong stepwise direction given.")
            return

    def _make_table(self):
        # @TODO: decide if should use random number for name & namespace
        self.models = session.table("stepwise", self.role)

    def _put_value(self, key, value):
        """
        wrapper to put key, value pair into models table
        """
        self.models.put(key, value)

    def _get_value(self, key):
        """
        wrapper to get value of a given key from models table 
        """
        return self.models.get(key)

    def _get_k(self):
        """
        Helper function only called by Arbiter, get the penalty coefficient for AIC/BIC calculation.
        """
        if self.score == "aic":
            self.k = 2
        elif self.score == "bic":
            self.k = np.log(self.n_count)
        else:
            raise ValueError("wrong score name given: {}. Only 'aic' or 'bic' acceptable.".format(self.score))

    def get_dfe(self, model, list1,list2):
        dfe = len(list1) + len(list2)
        if model.fit_intercept:
            dfe += 1
        return dfe

    def _get_step_best(self, step_models):
        best_score = -1
        best_model = ()
        for model in step_models:
            model_info = self.models_trained[model]
            score = model_info.get_score()
            if best_score < 0 or score < best_score:
                best_score = score
                best_model = model
        return best_model
           
    def drop_one(self, to_drop):
       for i in range(-1, len(to_drop)):
            dropped_list = list(to_drop)
            if i > -1:
                del dropped_list[i]
            yield dropped_list

    def add_one(self, to_add, original_list):
       for i in range(len(to_add)):
            added_list = original_list.append(to_add[i]).sort()
            yield added_list

    def _arbiter_run_step(self, model, host_list, guest_list, n_model):
        # run this step
        dfe = self.get_dfe(model, host_list, guest_list)

        host_feature_list = self.transfer_variable.host_feature_list
        guest_feature_list = self.transfer_variable.guest_feature_list

        host_feature_list.remote(host_list, idx=0)
        guest_feature_list.remote(host_list, idx=0)

        curr_step = Step()
        curr_step._set_step_info(self.step_direction, self.n_step, n_model)
        loss = curr_step.run(self.model_param, model, None, None, [])
        IC_computer = IC()
        IC_val = IC_computer.compute(self.k, self.n_count, dfe, loss)
        return IC_val
        
    def _arbiter_run(self, model):
        host_data_info = self.transfer_variable.host_data_info
        guest_data_info = self.transfer_variable.guest_data_info
        #host_feature_list = self.transfer_variable.host_feature_list
        #guest_feature_list = self.transfer_variable.guest_feature_list
        n_host, j_host = host_data_info.get(idx=0)
        n_guest, j_guest = guest_data_info.get(idx=0)
        self.n_count = n_host
        j = j_host + j_guest
        host_to_drop, guest_to_drop = list(range(j_host)), list(range(j_host))
        self._get_k()

        while self.n_step < self.max_step:
            step_models = set()
            if self.backward:
                n_model = 0
                self.step_direction = "backward"
                host_lists = self.drop_one(host_to_drop)
                guest_lists = self.drop_one(guest_to_drop)
                for host_list in host_lists:
                    host_tup, guest_tup = tuple(host_list), tuple(guest_to_drop)
                    # add models to current step set, check all models from step models at the end, use the feature
                    # lists as keys to get corresponding score
                    # from models_trained set; keep n_model as key maker to abstract out models at the end of stepwise
                    step_models.add((host_tup, guest_tup))
                    # skip this model if already trained and recorded
                    if (host_tup, guest_tup) in self.models_trained:
                        continue
                    IC_val = self._arbiter_run_step(model, host_list, guest_to_drop, n_model)
                    # store model criteria value in dict for future references
                    self.models_trained[(host_tup, guest_tup)] = ModelInfo(self.step_direction, self.n_step, n_model,
                                                                           IC_val)
                    n_model += 1
                    """
                    dfe = self.get_dfe(model, host_list, guest_to_drop)
                    host_feature_list.remote(host_list, idx=0)
                    guest_feature_list.remote(guest_to_drop, idx=0)
                    curr_step = Step()
                    curr_step._set_step_info(self.step_direction, self.n_step, n_model)
                    loss = curr_step.run(self.model_param, model, None, None, [])
                    IC_computer = IC()
                    IC_val = IC_computer.compute(self.k, self.n_count, dfe, loss)
                    # store model criteria value in dict for future references
                    self.models_trained[(host_tup, guest_tup)] = ModelInfo(self.step_direction, self.n_step, n_model, IC_val)
                    """
                for guest_list in guest_lists:
                    host_tup, guest_tup = tuple(host_to_drop), tuple(guest_list)
                    step_models.add((host_tup, guest_tup))
                    if (host_tup, guest_tup) in self.models_trained:
                        continue
                    IC_val = self._arbiter_run_step(model, host_to_drop, guest_list, n_model)
                    self.models_trained[(host_tup, guest_tup)] = ModelInfo(self.step_direction, self.n_step, n_model,
                                                                           IC_val)
                    n_model += 1
            if self.forward:
                n_model = 0
                self.step_direction = "forward"
                if self.n_step == 0 and len(self.models_trained) == 0:
                    # @TODO: initialize initial lists of one variable from each role
                    host_list = [0]
                    guest_list = [0]
                

            # @TODO: select the best model based on criteria value, update to_drop & to_add lists
            host_step_best, guest_step_best = self._get_step_best(step_models)
            self.n_step += 1
        # @TODO: arbiter sends the best model lists to Host & Guest (use H & G lists transfer variable: guest/host feature_list)
        # @TODO: make sure table should be manually destroyed
        self.models.destroy()

    def run(self, component_parameters, train_data, test_data, model):
        self._init_model(component_parameters)
        if self.forward:
            self.step_direction = "forward"
        if self.role == consts.ARBITER:
            self._arbiter_run(model)
        elif self.role == consts.HOST:
            host_data_info = self.transfer_variable.host_data_info
            n, j = train_data.count(), data_overview.get_features_shape(train_data)
            host_data_info.remote((n, j), idx=0)
        elif self.role == consts.GUEST:
            guest_data_info = self.transfer_variable.guest_data_info
            n, j = train_data.count(), data_overview.get_features_shape(train_data)
            guest_data_info.remote((n, j), idx=0)

        # @TODO: at each model, initialize step and call set_step_info, then step.run() to train model & predict
        # @TODO: drop_one & add_one for each step
        # @TODO use "map" to make new dTable








