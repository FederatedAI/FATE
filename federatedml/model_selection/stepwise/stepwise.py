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


from arch.api.utils import log_utils
from federatedml.evaluation.evaluation import IC
from federatedml.statistic import data_overview
from federatedml.model_selection.stepwise.step import Step
from federatedml.transfer_variable.transfer_class.stepwise_transfer_variable import StepwiseTransferVariable
from federatedml.util import consts

import itertools

import numpy as np

LOGGER = log_utils.getLogger()


class Stepwise(object):

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
        self.score = param.score
        self.direction = param.direction
        self.p_enter = param.p_enter
        self.p_remove = param.p_remove
        self.max_step = param.max_step
        self.score = param.score
        self._get_direction()

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

    def _get_dfe(self, host_list, guest_list):
        if self.mode == consts.HETERO:
            return len(host_list) + len(guest_list)
        elif self.mode == consts.HOMO:
            assert(len(host_list) == guest_list), "Host & Guest have receive feature lists of different lengths under HOMO mode."
            return len(host_list)
        else:
            LOGGER.warn("Unknown mode {}. Must be one of 'HETERO' or 'HOMO' ".format(self.mode))

    def dropone_list(self, host_list, guest_list):
        host_lists, guest_lists = [], []
        for i in range(len(host_list)):
            new_host_list = list(host_list)
            del new_host_list[i]
            host_lists.append(new_host_list)
            guest_lists.append(guest_list)
        for i in range(len(host_list)):
            new_guest_list = list(guest_list)
            del new_guest_list[i]
            guest_lists.append(new_guest_list)
            host_lists.append(host_list)

        return host_lists, guest_lists

    def drop_one(self, to_drop):
       for i in range(len(to_drop)):
            dropped_list = list(to_drop)
            del dropped_list[i]
            yield dropped_list

    def add_one(self, to_add, original_list):
       for i in range(len(to_add)):
            added_list = original_list.append(to_add[i]).sort()
            yield added_list

    def _arbiter_run(self, model):
        transfer_variable = StepwiseTransferVariable()
        host_data_info = transfer_variable.host_data_info
        guest_data_info = transfer_variable.guest_data_info
        host_feature_list = transfer_variable.host_feature_list
        guest_feature_list = transfer_variable.guest_feature_list
        n_host, j_host = host_data_info.get(idx=0)
        n_guest, j_guest = guest_data_info.get(idx=0)
        if self.mode == consts.HOMO:
            self.n_count = n_host + n_guest
            j = j_host
        elif self.mode == consts.HETERO:
            self.n_count = n_host
            j = j_host + j_guest
        else:
            LOGGER.warn("Unknwon mode {} for stepwise.".format(self.mode))
        self._get_k()

        while self.n_step < self.max_step:
            if self.backward:
                n_model = 0
                if self.n_step == 0 and n_model == 0:
                    #@TODO: decide if full list should be step 0 or before
                    host_lists = [list(range(j_host))]
                    guest_lists = [list(range(j_guest))]
                else:
                    if self.mode == consts.HETERO:
                        host_list, guest_list = list(range(j_host)), list(range(j_host))
                        host_list_generator = self.drop_one(host_list)
                        guest_list_generator = self.drop_one(guest_list)
                        for host_list in host_list_generator:
                            dfe = self._get_dfe(host_list, guest_list)
                            host_feature_list.remote(host_list, idx=0)
                            guest_feature_list.remote(guest_list, idx=0)
                            curr_step = Step()
                            curr_step._set_step_info(self.step_direction, self.n_step, n_model)
                            loss = curr_step.run(self.model_param, model, None, None, [])
                            IC_computer = IC()
                            if model.param.fit_intercept:
                                dfe += 1
                            IC_val = IC_computer.compute(self.k, self.n_count, dfe, loss)
                        for guest_list in guest_list_generator:
                            dfe = self._get_dfe(host_list, guest_list)
                            host_feature_list.remote(host_list, idx=0)
                            guest_feature_list.remote(guest_list, idx=0)
                            curr_step = Step()
                            curr_step._set_step_info(self.step_direction, self.n_step, n_model)
                            loss = curr_step.run(self.model_param, model, None, None, [])
                            IC_computer = IC()
                            if model.param.fit_intercept:
                                dfe += 1
                            IC_val = IC_computer.compute(self.k, self.n_count, dfe, loss)
                            n_model += 1
            if self.forward:
                if self.n_step == 0 and n_model == 0:
                    host_list = [0]
                    guest_list = [0]

            self.n_step += 1

    def run(self, component_parameters, train_data, test_data, model):
        self._init_model(component_parameters)
        if self.forward:
            self.step_direction = "forward"
        if self.role == consts.ARBITER:
            self._arbiter_run(model)
        elif self.role == consts.HOST:
            transfer_variable = StepwiseTransferVariable()
            host_data_info = transfer_variable.host_data_info
            n, j = train_data.count(), data_overview.get_features_shape(train_data)
            host_data_info.remote((n, j), idx=0)
        elif self.role == consts.GUEST:
            transfer_variable = StepwiseTransferVariable()
            guest_data_info = transfer_variable.guest_data_info
            n, j = train_data.count(), data_overview.get_features_shape(train_data)
            guest_data_info.remote((n, j), idx=0)

        # @TODO: at each model, initialize step and call set_step_info, then step.run() to train model & predict
        # @TODO: drop_one & add_one for each step
        # @TODO use "map" to make new dTable








