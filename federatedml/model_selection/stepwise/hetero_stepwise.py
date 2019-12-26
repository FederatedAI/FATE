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

import copy
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
        return (self.step_direction, self.n_step, self.n_model)

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
        self.stop_stepwise = False
        self.models = None

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
        self.make_table()
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
            raise ValueError("Wrong stepwise direction given.")

    def make_table(self):
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

    def _set_k(self):
        """
        Helper function only called by Arbiter, get the penalty coefficient for AIC/BIC calculation.
        """
        if self.score == "aic":
            self.k = 2
        elif self.score == "bic":
            self.k = np.log(self.n_count)
        else:
            raise ValueError("wrong score name given: {}. Only 'aic' or 'bic' acceptable.".format(self.score))

    @staticmethod
    def get_dfe_list(model, list1, list2):
        dfe = len(list1) + len(list2)
        if model.fit_intercept:
            dfe += 1
        return dfe

    @staticmethod
    def get_dfe(model, mask1, mask2):
        dfe = sum(mask1) + sum(mask2)
        if model.fit_intercept:
            dfe += 1
        return dfe

    def get_step_best(self, step_models):
        best_score = -1
        best_model = ()
        for model in step_models:
            model_info = self.models_trained[model]
            score = model_info.get_score()
            if best_score < 0 or score < best_score:
                best_score = score
                best_model = model
        return best_model
    '''
    # @TODO: improve efficiency
    def drop_one_list(self, host_list, guest_list):
        host_lists, guest_lists = [], []
        for i in range(len(host_list)):
            new_host_list = list(host_list)
            del new_host_list[i]
            if len(new_host_list) == 0:
                break
            host_lists.append(new_host_list)
            guest_lists.append(guest_list)
        for i in range(len(host_list)):
            new_guest_list = list(guest_list)
            del new_guest_list[i]
            if len(new_guest_list) == 0:
                break
            guest_lists.append(new_guest_list)
            host_lists.append(host_list)
        return host_lists, guest_lists

    def add_one_list(self, host_list, guest_list, host_to_add, guest_to_add):
        host_lists, guest_lists = [], []
        # initial step
        if len(host_list) == 0 and len(guest_list) == 0:
            p = list(itertools.product(host_to_add, guest_to_add))
            host_lists = [[x[0]] for x in p]
            guest_lists = [[x[1]] for x in p]
            return host_lists, guest_lists
        for feature in host_to_add:
            new_host_list = list(host_list)
            new_host_list.append(feature)
            new_host_list.sort()
            host_lists.append(new_host_list)
            guest_lists.append(guest_list)
        for feature in guest_to_add:
            new_guest_list = list(guest_list)
            new_guest_list.append(feature)
            new_guest_list.sort()
            guest_lists.append(new_guest_list)
            host_lists.append(host_list)
        return host_lists, guest_lists
    '''
    def drop_one(self, host_mask, guest_mask):
        host_masks, guest_masks = [], []
        for i in np.where(host_mask > 0)[0]:
            new_host_mask = np.copy(host_mask)
            new_host_mask[i] = 0
            if sum(new_host_mask) == 0:
                break
            host_masks.append(new_host_mask)
            guest_masks.append(np.copy(guest_mask))
        for i in np.where(guest_mask > 0)[0]:
            new_guest_mask = np.copy(guest_mask)
            new_guest_mask[i] = 0
            if sum(new_guest_mask) == 0:
                break
            guest_masks.append(new_guest_mask)
            host_masks.append(np.copy(host_mask))
        return host_masks, guest_masks

    def add_one(self, host_mask, guest_mask):
        host_masks, guest_masks = [], []
        # initial step
        if sum(host_mask) == 0 and sum(guest_mask) == 0:
            host_masks = np.eye(host_mask.size, host_mask.size)
            host_masks = np.repeat(host_masks, guest_mask.size, axis=0)
            guest_masks = np.eye(guest_mask.size, guest_mask.size)
            guest_masks = np.tile(guest_masks, (host_mask.size, 1))
            return list(host_masks), list(guest_masks)

        for i in np.where(host_mask < 1):
            new_host_mask = np.copy(host_mask)
            new_host_mask[i] = 1
            host_masks.append(new_host_mask)
            guest_masks.append(np.copy(guest_mask))
        for i in np.where(guest_mask < 1)[0]:
            new_guest_mask = np.copy(guest_mask)
            new_guest_mask[i] = 0
            guest_masks.append(new_guest_mask)
            host_masks.append(np.copy(host_mask))
        return host_masks, guest_masks
    '''
    def filter_feature_lists(self, host_lists, guest_lists):
        filtered_host_lists, filtered_guest_lists = map(list, zip(*([h, g] for h, g in zip(host_lists, guest_lists)
                                                                    if (tuple(h), tuple(g)) not in self.models_trained)))
        return filtered_host_lists, filtered_guest_lists

    def record_step_models_list(self, step_models, host_lists, guest_lists):
        for model_list in list((tuple(h), tuple(g)) for h, g in zip(host_lists, guest_lists)):
            step_models.add(model_list)
    '''

    def filter_feature_masks(self, host_masks, guest_masks):
        filtered_host_masks, filtered_guest_masks = map(list, zip(*((h, g) for h, g in zip(host_masks, guest_masks)
                                                                    if (tuple(h), tuple(g)) not in self.models_trained)))
        return filtered_host_masks, filtered_guest_masks

    def record_step_models(self, step_models, host_masks, guest_masks):
        for model_mask in list((tuple(h), tuple(g)) for h, g in zip(host_masks, guest_masks)):
            step_models.add(model_mask)

    def get_new_to_add(self, new_to_drop, k_count):
        new_to_add = list(filter(lambda x: x not in range(k_count), new_to_drop))
        return new_to_add
    '''
    def check_best_list(self, new_host_to_drop, new_guest_to_drop, host_to_drop, guest_to_drop, j_host, j_guest):
        # if model not updated
        if new_host_to_drop == host_to_drop and new_guest_to_drop == guest_to_drop:
            return True
        # if full model is the best
        elif len(new_guest_to_drop) == j_host and len(new_guest_to_drop) == j_guest:
            return True
        return False
    '''
    def check_best(self, new_host_mask, new_guest_mask, host_mask, guest_mask, j_host, j_guest):
        # if model not updated
        if new_host_mask == host_mask and new_guest_mask == guest_mask:
            return True
        # if full model is the best
        elif sum(new_guest_mask) == j_host and sum(new_guest_mask) == j_guest:
            return True
        return False

    def _arbiter_run_step(self, model, host_mask, guest_mask, n_model):
        dfe = self.get_dfe(model, host_mask, guest_mask)
        current_key = (self.step_direction, self.n_step, n_model)
        current_step = Step()
        current_step.set_step_info(current_key)
        trained_model = current_step.run(model, None, None, None)
        # get final loss from loss history for criteria calculation
        loss = trained_model.loss_history[-1]
        IC_computer = IC()
        ic_val = IC_computer.compute(self.k, self.n_count, dfe, loss)
        if np.isinf(ic_val):
            raise ValueError("Loss value of infinity obtained. Stepwise stopped.")
        host_tup, guest_tup = tuple(host_mask), tuple(guest_mask)
        self.models_trained[(host_tup, guest_tup)] = ModelInfo(self.step_direction, self.n_step, n_model,
                                                               ic_val)
        current_key = self.make_key(self.step_direction, self.n_step, n_model)
        self._put_value(current_key, trained_model)

    def arbiter_sync_stop_stepwise(self):
        self.transfer_variable.stop_stepwise.remote(self.stop_stepwise, role=consts.HOST, suffix=(self.n_step,))
        self.transfer_variable.stop_stepwise.remote(self.stop_stepwise, role=consts.GUEST, suffix=(self.n_step,))

    def client_sync_stop_stepwise(self):
        self.stop_stepwise = self.transfer_variable.stop_stepwise.get(suffix=(self.n_step,))
    '''
    def arbiter_sync_step_info_list(self, host_lists, guest_lists):
        self.transfer_variable.host_step_info.remote((self.step_direction, self.n_step, host_lists))
        self.transfer_variable.guest_step_info.remote((self.step_direction, self.n_step, guest_lists))
    '''
    def arbiter_sync_step_info(self, host_masks, guest_masks):
        self.transfer_variable.host_step_info.remote((self.step_direction, self.n_step, host_masks))
        self.transfer_variable.guest_step_info.remote((self.step_direction, self.n_step, guest_masks))

    def client_sync_step_info(self):
        if self.role == consts.HOST:
            return self.transfer_variable.host_step_info.get()
        elif self.role == consts.GUEST:
            return self.transfer_variable.guest_step_info.get()
        else:
            raise ValueError("unknown role {} encountered!".format(self.role))

    def make_key(self, step_direction, n_step, n_model):
        """
        make key for models table
        """
        return (step_direction, n_step, n_model)
    '''
    def _arbiter_run_list(self, model):
        n_host, j_host = self.host_data_info_transfer.get(idx=0)
        n_guest, j_guest = self.guest_data_info_transfer.get(idx=0)
        self.n_count = n_host
        if self.backward:
            host_to_drop, guest_to_drop = list(range(j_host)), list(range(j_guest))
            host_to_add, guest_to_add = [], []
        else:
            host_to_drop, guest_to_drop = [], []
            host_to_add, guest_to_add = list(range(j_host)), list(range(j_guest))
        self._set_k()
        while self.n_step < self.max_step:
            step_models = set()
            n_model = 0
            if self.backward:
                self.step_direction = "backward"
                host_lists, guest_lists = self.drop_one(host_to_drop, guest_to_drop)
                # add all models of current step into step_models
                self.record_step_models(step_models, host_lists, guest_lists)
                # filter out models already trained
                host_lists, guest_lists = self.filter_feature_lists(host_lists, guest_lists)
                self.arbiter_sync_step_info(host_lists, guest_lists)
                for i in range(len(host_lists)):
                    host_list, guest_list = host_lists[i], guest_lists[i]
                    self._arbiter_run_step(model, host_list, guest_list, n_model)
                    n_model += 1
            if self.forward:
                n_model = 0
                self.step_direction = "forward"
                host_lists, guest_lists = self.add_one(host_to_drop, guest_to_drop, host_to_add, guest_to_add)
                # add all models of current step into step_models
                self.record_step_models(step_models, host_lists, guest_lists)
                # filter out models already trained
                host_lists, guest_lists = self.filter_feature_lists(host_lists, guest_lists)
                self.arbiter_sync_step_info(host_lists, guest_lists)
                for i in range(len(host_lists)):
                    host_list, guest_list = host_lists[i], guest_lists[i]
                    self._arbiter_run_step(model, host_list, guest_list, n_model)
                    n_model += 1
            new_host_to_drop_tup, new_guest_to_drop_tup = self.get_step_best(step_models)
            new_host_to_drop, new_guest_to_drop = list(new_host_to_drop_tup), list(new_guest_to_drop_tup)
            host_to_add = self.get_new_to_add(new_host_to_drop, j_host)
            guest_to_add = self.get_new_to_add(new_guest_to_drop, j_guest)
            self.stop_stepwise = self.check_best(new_host_to_drop, new_guest_to_drop,
                                                 host_to_drop, guest_to_drop, j_host, j_guest)
            self.arbiter_sync_stop_stepwise()
            # for prettify output format, skip add extra 1 for self.n_step
            if self.stop_stepwise:
                break
            host_to_drop, guest_to_drop = new_host_to_drop, new_guest_to_drop
            self.n_step += 1

        best_model = self.models_trained[(tuple(host_to_drop), tuple(guest_to_drop))]
        best_model_key = best_model.get_key()
        self.transfer_variable.host_best_model.remote(best_model_key, idx=0)
        self.transfer_variable.guest_best_model.remote(best_model_key, idx=0)
        best_model = copy.deepcopy(self._get_value(best_model_key))
        model.load_model(best_model)
        self.models.destroy()
    '''
    def _arbiter_run(self, model):
        n_host, j_host = self.host_data_info_transfer.get(idx=0)
        n_guest, j_guest = self.guest_data_info_transfer.get(idx=0)
        self.n_count = n_host
        if self.backward:
            host_mask, guest_mask = np.ones(j_host), np.ones(j_guest)
        else:
            host_mask, guest_mask = np.zeros(j_host), np.zeros(j_guest)
        self._set_k()
        while self.n_step < self.max_step:
            step_models = set()
            n_model = 0
            if self.backward:
                self.step_direction = "backward"
                host_masks, guest_masks = self.drop_one(host_mask, guest_mask)
                # add all models of current step into step_models
                self.record_step_models(step_models, host_masks, guest_masks)
                # filter out models already trained
                host_masks, guest_masks = self.filter_feature_masks(host_masks, guest_masks)
                self.arbiter_sync_step_info(host_masks, guest_masks)
                for i in range(len(host_masks)):
                    host_mask, guest_mask = host_masks[i], guest_masks[i]
                    self._arbiter_run_step(model, host_mask, guest_mask, n_model)
                    n_model += 1
            if self.forward:
                n_model = 0
                self.step_direction = "forward"
                host_masks, guest_masks = self.add_one(host_mask, guest_mask)
                # add all models of current step into step_models
                self.record_step_models(step_models, host_masks, guest_masks)
                # filter out models already trained
                host_masks, guest_masks = self.filter_feature_masks(host_masks, guest_masks)
                self.arbiter_sync_step_info(host_masks, guest_masks)
                for i in range(len(host_masks)):
                    host_mask, guest_mask = host_masks[i], guest_masks[i]
                    self._arbiter_run_step(model, host_mask, guest_mask, n_model)
                    n_model += 1
            new_host_mask_tup, new_guest_mask_tup = self.get_step_best(step_models)
            new_host_mask, new_guest_mask = np.array(new_host_mask_tup), np.array(new_guest_mask_tup)
            #host_to_add = self.get_new_to_add(new_host_to_drop, j_host)
            #guest_to_add = self.get_new_to_add(new_guest_to_drop, j_guest)
            self.stop_stepwise = self.check_best(new_host_mask, new_guest_mask,
                                                 host_mask, guest_mask, j_host, j_guest)
            self.arbiter_sync_stop_stepwise()
            # for prettify output format, skip add extra 1 for self.n_step
            if self.stop_stepwise:
                break
            host_mask, guest_mask = new_host_mask, new_guest_mask
            self.n_step += 1

        best_model = self.models_trained[(tuple(host_mask), tuple(guest_mask))]
        best_model_key = best_model.get_key()
        self.transfer_variable.host_best_model.remote(best_model_key, idx=0)
        self.transfer_variable.guest_best_model.remote(best_model_key, idx=0)
        best_model = copy.deepcopy(self._get_value(best_model_key))
        model.load_model(best_model)
        self.models.destroy()
    '''
    def run_list(self, component_parameters, train_data, test_data, model):
        self._init_model(component_parameters)
        if self.role == consts.ARBITER:
            self._arbiter_run(model)
            return
        n, j = train_data.count(), data_overview.get_features_shape(train_data)
        if self.role == consts.HOST:
            self.host_data_info_transfer = self.transfer_variable.host_data_info
            self.host_data_info_transfer.remote((n, j), idx=0)
        elif self.role == consts.GUEST:
            self.guest_data_info_transfer = self.transfer_variable.guest_data_info
            self.guest_data_info_transfer.remote((n, j), idx=0)
        while not self.stop_stepwise:
            step_direction, n_step, feature_lists = self.client_sync_step_info()
            self.step_direction, self.n_step = step_direction, n_step
            n_model = 0
            for feature_list in feature_lists:
                current_key = self.make_key(self.step_direction, self.n_step, n_model)
                current_step = Step()
                current_step.set_step_info(current_key)
                model = current_step.run(model, train_data, test_data, feature_list)
                self._put_value(current_key, model)
                n_model += 1
            self.client_sync_stop_stepwise()
            self.n_step += 1
        if self.role == consts.HOST:
            best_model_key = self.transfer_variable.host_best_model.get()
        else:
            best_model_key = self.transfer_variable.guest_best_model.get()
        # @TODO: use load_model to copy information from best model to original model; need to decide if best model need to be deep copied
        best_model = copy.deepcopy(self._get_value(best_model_key))
        model.load_model(best_model)
        self.models.destroy()
    '''
    def run(self, component_parameters, train_data, test_data, model):
        self._init_model(component_parameters)
        if self.role == consts.ARBITER:
            self._arbiter_run(model)
            return
        n, j = train_data.count(), data_overview.get_features_shape(train_data)
        if self.role == consts.HOST:
            self.host_data_info_transfer = self.transfer_variable.host_data_info
            self.host_data_info_transfer.remote((n, j), idx=0)
        elif self.role == consts.GUEST:
            self.guest_data_info_transfer = self.transfer_variable.guest_data_info
            self.guest_data_info_transfer.remote((n, j), idx=0)
        while not self.stop_stepwise:
            step_direction, n_step, feature_masks = self.client_sync_step_info()
            self.step_direction, self.n_step = step_direction, n_step
            n_model = 0
            for feature_mask in feature_masks:
                current_key = self.make_key(self.step_direction, self.n_step, n_model)
                current_step = Step()
                current_step.set_step_info(current_key)
                model = current_step.run(model, train_data, test_data, feature_mask)
                self._put_value(current_key, model)
                n_model += 1
            self.client_sync_stop_stepwise()
            self.n_step += 1
        if self.role == consts.HOST:
            best_model_key = self.transfer_variable.host_best_model.get()
        else:
            best_model_key = self.transfer_variable.guest_best_model.get()
        # @TODO: use load_model to copy information from best model to original model; need to decide if best model need to be deep copied
        best_model = copy.deepcopy(self._get_value(best_model_key))
        model.load_model(best_model)
        self.models.destroy()
