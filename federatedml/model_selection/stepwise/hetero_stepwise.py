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
from federatedml.model_selection.stepwise.step import Step
from federatedml.statistic import data_overview
from federatedml.transfer_variable.transfer_class.stepwise_transfer_variable import StepwiseTransferVariable
from federatedml.util import consts

import numpy as np

LOGGER = log_utils.getLogger()
session.init("stepwise")


class ModelInfo(object):
    def __init__(self, n_step, n_model, score):
        self.score = score
        self.n_step = n_step
        self.n_model = n_model

    def get_key(self):
        return (self.n_step, self.n_model)

    def get_score(self):
        return self.score


class HeteroStepwise(object):
    def __init__(self):
        self.mode = None
        self.role = None
        self.forward = False
        self.backward = False
        self.best_list = []
        self.n_step = -1
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
        self.max_step = param.max_step
        self.nvmin = param.nvmin
        self.nvmax = param.nvmax
        self.transfer_variable = StepwiseTransferVariable()
        self.host_data_info_transfer = self.transfer_variable.host_data_info
        self.guest_data_info_transfer = self.transfer_variable.guest_data_info
        self._get_direction()
        self.make_table()
        # only used by Arbiter to fast filter model
        self.models_trained = {}
        # only used by Arbiter to calculate criteria value
        self.IC_computer = None

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
        model = self.models.get(key)
        return model

    def _set_k(self):
        """
        Helper function only called by Arbiter, get the penalty coefficient for AIC/BIC calculation.
        """
        if self.score_name == "aic":
            self.k = 2
        elif self.score_name == "bic":
            self.k = np.log(self.n_count)
        else:
            raise ValueError("Wrong score name given: {}. Only 'aic' or 'bic' acceptable.".format(self.score_name))

    @staticmethod
    def get_dfe(model, mask1, mask2):
        dfe = sum(mask1) + sum(mask2)
        if model.fit_intercept:
            dfe += 1
            LOGGER.debug("fit_intercept detected, 1 is added to dfe")
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

    def drop_one(self, host_mask, guest_mask):
        host_masks, guest_masks = [], []
        # initial step, add full model to comparison
        if sum(host_mask) == host_mask.size and sum(guest_mask) == guest_mask.size:
            host_masks.append(np.copy(host_mask))
            guest_masks.append(np.copy(guest_mask))
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
        LOGGER.debug("step {}, in drop_one, host_masks len: {}, guests_masks len: {}".format(self.n_step,
                                                                                             len(host_masks),
                                                                                             len(guest_masks)))
        LOGGER.debug("step {}, in drop_one, host_masks: {}, guest_masks: {} ".format(self.n_step,
                                                                                     host_masks, guest_masks))
        return host_masks, guest_masks

    def add_one(self, host_mask, guest_mask):
        host_masks, guest_masks = [], []
        # initial masks, one feature on each side
        if sum(host_mask) == 0 and sum(guest_mask) == 0:
            host_masks = np.eye(host_mask.size, host_mask.size, dtype=bool)
            host_masks = np.repeat(host_masks, guest_mask.size, axis=0)
            guest_masks = np.eye(guest_mask.size, guest_mask.size, dtype=bool)
            guest_masks = np.tile(guest_masks, (host_mask.size, 1))
            LOGGER.debug("step {}, in add_one, host_masks len: {}, guests_masks len: {}".format(self.n_step,
                                                                                                len(host_masks),
                                                                                                len(guest_masks)))
            LOGGER.debug("step {}, in add_one, host_masks: {}, guests_masks: {}".format(self.n_step,
                                                                                        host_masks, guest_masks))
            return list(host_masks), list(guest_masks)

        for i in np.where(host_mask < 1)[0]:
            new_host_mask = np.copy(host_mask)
            new_host_mask[i] = 1
            host_masks.append(new_host_mask)
            guest_masks.append(np.copy(guest_mask))
        for i in np.where(guest_mask < 1)[0]:
            new_guest_mask = np.copy(guest_mask)
            new_guest_mask[i] = 1
            guest_masks.append(new_guest_mask)
            host_masks.append(np.copy(host_mask))
        LOGGER.debug("step {}, in add_one, host_masks len: {}, guests_masks len: {}".format(self.n_step,
                                                                                            len(host_masks),
                                                                                            len(guest_masks)))
        LOGGER.debug("step {}, in add_one, host_masks: {}, guests_masks: {}".format(self.n_step,
                                                                                    host_masks, guest_masks))
        return host_masks, guest_masks

    def filter_feature_masks(self, host_masks, guest_masks):
        filtered_host_masks, filtered_guest_masks = map(list, zip(*((h, g) for h, g in zip(host_masks, guest_masks)
                                                                    if
                                                                    (tuple(h), tuple(g)) not in self.models_trained)))
        count_diff = len(host_masks) - len(filtered_host_masks)
        LOGGER.debug("step {}, filtered {} masks".format(self.n_step, count_diff))
        return filtered_host_masks, filtered_guest_masks

    @staticmethod
    def record_step_models(step_models, host_masks, guest_masks):
        for model_mask in list((tuple(h), tuple(g)) for h, g in zip(host_masks, guest_masks)):
            step_models.add(model_mask)

    def check_stop(self, new_host_mask, new_guest_mask, host_mask, guest_mask):
        # if model not updated
        if np.array_equal(new_host_mask, host_mask) and np.array_equal(new_guest_mask, guest_mask):
            LOGGER.debug("masks not changed, check_stop returns True")
            return True
        # if full model is the best
        if sum(new_host_mask < 1) == 0 and sum(new_guest_mask < 1) == 0:
            LOGGER.debug("masks are full model, check_stop returns True")
            return True
        # if new best reach variable count lower limit
        new_total_nv = sum(new_host_mask) + sum(new_guest_mask)
        total_nv = sum(host_mask) + sum(guest_mask)
        if new_total_nv == self.nvmin and total_nv >= self.nvmin:
            LOGGER.debug("variable count min reached, check_stop returns True")
            return True
        # if new best reach variable count upper limit
        if self.nvmax is not None:
            if new_total_nv == self.nvmax and total_nv <= self.nvmax:
                LOGGER.debug("variable count max reached, check_stop returns True")
                return True
        # if reach max step
        if self.n_step == self.max_step - 1:
            LOGGER.debug("max step reached, check_stop returns True")
            return True
        return False

    def _arbiter_run_step(self, model, host_mask, guest_mask, n_model):
        dfe = HeteroStepwise.get_dfe(model, host_mask, guest_mask)
        current_key = (self.n_step, n_model)
        current_step = Step()
        current_step.set_step_info(current_key)
        trained_model = current_step.run(model, None, None, None)
        if len(trained_model.loss_history) == 0:
            raise ValueError("Arbiter has no loss history. Stepwise does not support model without total loss.")
        # get final loss from loss history for criteria calculation
        loss = trained_model.loss_history[-1]
        ic_val = self.IC_computer.compute(self.k, self.n_count, dfe, loss)
        LOGGER.info("step {} n_model {}: ic_val {}".format(self.n_step, n_model, ic_val))
        if np.isinf(ic_val):
            raise ValueError("Loss value of infinity obtained. Stepwise stopped.")
        host_tup, guest_tup = tuple(host_mask), tuple(guest_mask)
        self.models_trained[(host_tup, guest_tup)] = ModelInfo(self.n_step, n_model, ic_val)
        current_key = HeteroStepwise.make_key(self.n_step, n_model)
        self._put_value(current_key, trained_model)

    def arbiter_sync_stop_stepwise(self):
        self.transfer_variable.stop_stepwise.remote(self.stop_stepwise, role=consts.HOST, suffix=(self.n_step,))
        LOGGER.info("Arbiter remotes step {} stop_stepwise {} to Host.".format(self.n_step, self.stop_stepwise))
        self.transfer_variable.stop_stepwise.remote(self.stop_stepwise, role=consts.GUEST, suffix=(self.n_step,))
        LOGGER.info("Arbiter remotes step {} stop_stepwise {} to Guest.".format(self.n_step, self.stop_stepwise))

    def client_sync_stop_stepwise(self):
        self.stop_stepwise = self.transfer_variable.stop_stepwise.get(suffix=(self.n_step,))[0]
        LOGGER.info("{} receives stop_stepwise {} from Arbiter.".format(self.role, self.stop_stepwise))

    def arbiter_sync_step_info(self, host_masks, guest_masks):
        self.transfer_variable.host_step_info.remote(host_masks,
                                                     suffix=(self.n_step,))
        LOGGER.info("Arbiter remotes step info {} to Host.".format(self.n_step))
        self.transfer_variable.guest_step_info.remote(guest_masks,
                                                      suffix=(self.n_step,))
        LOGGER.info("Arbiter remotes step info {} to Guest.".format(self.n_step))

    def client_sync_step_info(self):
        if self.role == consts.HOST:
            LOGGER.info("Host receives step info {} from Arbiter.".format(self.n_step))
            return self.transfer_variable.host_step_info.get(suffix=(self.n_step,))
        elif self.role == consts.GUEST:
            LOGGER.info("Guest receives step info {} from Arbiter.".format(self.n_step))
            return self.transfer_variable.guest_step_info.get(suffix=(self.n_step,))
        else:
            raise ValueError("unknown role {} encountered!".format(self.role))

    @staticmethod
    def load_best_model(model, best_model):
        # put best_model into dict format so it can be loaded into original model
        model_dict = {'model': {'stepwise': best_model.export_model()}}
        model.load_model(model_dict)

    @staticmethod
    def make_key(n_step, n_model):
        return n_step, n_model

    @staticmethod
    def predict(data_instances, model):
        if data_instances is None:
            return
        d_header = data_instances.schema.get("header")
        best_feature = [d_header.index(x) for x in model.header]
        best_mask = np.zeros(len(d_header), dtype=bool)
        np.put(best_mask, best_feature, 1)
        new_data = data_instances.mapValues(lambda v: Step.slice_data_instance(v, best_mask))
        pred_result = model.predict(new_data)
        return pred_result

    def _arbiter_run(self, model):
        self.IC_computer = IC()
        n_host, j_host = self.host_data_info_transfer.get(idx=0)
        n_guest, j_guest = self.guest_data_info_transfer.get(idx=0)
        self.n_count = n_host
        if self.backward:
            host_mask, guest_mask = np.ones(j_host, dtype=bool), np.ones(j_guest, dtype=bool)
        else:
            host_mask, guest_mask = np.zeros(j_host, dtype=bool), np.zeros(j_guest, dtype=bool)
        self._set_k()
        while not self.stop_stepwise:
            self.n_step += 1
            LOGGER.info("Enter step {}".format(self.n_step))
            step_models = set()
            host_masks, guest_masks = [], []
            n_model = 0
            if self.backward:
                self.step_direction = "backward"
                LOGGER.info("step {}, direction: {}".format(self.n_step, self.step_direction))
                back_host_masks, back_guest_masks = self.drop_one(host_mask, guest_mask)
                host_masks = host_masks + back_host_masks
                guest_masks = guest_masks + back_guest_masks

            if (self.forward and self.n_step > 0) or (not self.backward):
                self.step_direction = "forward"
                LOGGER.info("step {}, direction: {}".format(self.n_step, self.step_direction))
                forward_host_masks, forward_guest_masks = self.add_one(host_mask, guest_mask)
                host_masks = host_masks + forward_host_masks
                guest_masks = guest_masks + forward_guest_masks

            HeteroStepwise.record_step_models(step_models, host_masks, guest_masks)
            host_masks, guest_masks = self.filter_feature_masks(host_masks, guest_masks)
            self.arbiter_sync_step_info(host_masks, guest_masks)
            for i in range(len(host_masks)):
                curr_host_mask, curr_guest_mask = host_masks[i], guest_masks[i]
                self._arbiter_run_step(model, curr_host_mask, curr_guest_mask, n_model)
                n_model += 1
            new_host_mask_tup, new_guest_mask_tup = self.get_step_best(step_models)
            new_host_mask, new_guest_mask = np.array(new_host_mask_tup), np.array(new_guest_mask_tup)
            LOGGER.debug(
                "step {}'s best host_mask {}, best guest_mask {}".format(self.n_step, new_host_mask, new_guest_mask))
            self.stop_stepwise = self.check_stop(new_host_mask, new_guest_mask, host_mask, guest_mask)
            host_mask, guest_mask = new_host_mask, new_guest_mask
            LOGGER.debug("step {} stop_stepwise is: {}".format(self.n_step, self.stop_stepwise))
            self.arbiter_sync_stop_stepwise()

        best_model = self.models_trained[(tuple(host_mask), tuple(guest_mask))]
        best_model_key = best_model.get_key()
        LOGGER.info("best_model_key is {}".format(best_model_key))
        self.transfer_variable.best_model.remote(best_model_key, role=consts.HOST, idx=0)
        self.transfer_variable.best_model.remote(best_model_key, role=consts.GUEST, idx=0)
        best_model = self._get_value(best_model_key)
        HeteroStepwise.load_best_model(model, best_model)
        self.models.destroy()

    def run(self, component_parameters, train_data, test_data, model):
        LOGGER.info("Enter stepwise")
        self._init_model(component_parameters)

        if self.role == consts.ARBITER:
            self._arbiter_run(model)
            return
        n, j = train_data.count(), data_overview.get_features_shape(train_data)
        if self.role == consts.HOST:
            self.host_data_info_transfer.remote((n, j), role=consts.ARBITER, idx=0)
        elif self.role == consts.GUEST:
            self.guest_data_info_transfer.remote((n, j), role=consts.ARBITER, idx=0)
        while not self.stop_stepwise:
            self.n_step += 1
            LOGGER.info("Enter step {}".format(self.n_step))
            feature_masks = self.client_sync_step_info()[0]
            n_model = 0
            for feature_mask in feature_masks:
                current_key = HeteroStepwise.make_key(self.n_step, n_model)
                current_step = Step()
                current_step.set_step_info(current_key)
                trained_model = current_step.run(model, train_data, test_data, feature_mask)
                self._put_value(current_key, trained_model)
                n_model += 1
            self.client_sync_stop_stepwise()

        best_model_key = self.transfer_variable.best_model.get()[0]
        best_model = self._get_value(best_model_key)
        HeteroStepwise.load_best_model(model, best_model)
        self.models.destroy()
