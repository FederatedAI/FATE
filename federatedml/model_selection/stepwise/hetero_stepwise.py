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
from fate_flow.entity.metric import Metric, MetricMeta
from federatedml.evaluation.evaluation import IC, IC_Approx
from federatedml.model_selection.stepwise.step import Step
from federatedml.statistic import data_overview
from federatedml.transfer_variable.transfer_class.stepwise_transfer_variable import StepwiseTransferVariable
from federatedml.util import consts

import itertools
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression, LinearRegression
from google.protobuf.json_format import MessageToDict
import uuid

LOGGER = log_utils.getLogger()


class ModelInfo(object):
    def __init__(self, n_step, n_model, score, loss, direction):
        self.score = score
        self.n_step = n_step
        self.n_model = n_model
        self.direction = direction
        self.loss = loss
        self.uid = str(uuid.uuid1())

    def get_score(self):
        return self.score

    def get_loss(self):
        return self.loss

    def get_key(self):
        return self.uid


class HeteroStepwise(object):
    def __init__(self):
        self.mode = None
        self.role = None
        self.forward = False
        self.backward = False
        self.n_step = 0
        self.has_test = False
        self.n_count = 0
        self.stop_stepwise = False
        self.models = None
        self.metric_namespace = "train"
        self.metric_type = "STEPWISE"
        self.intercept = None
        self.models = {}
        self.models_trained = {}
        self.IC_computer = None
        self.step_direction = None

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
            raise ValueError("Wrong stepwise direction given.")

    def _put_model(self, key, model):
        """
        wrapper to put key, model dict pair into models dict
        """
        model_dict = {'model': {'stepwise': model.export_model()}}
        self.models[key] = model_dict

    def _get_model(self, key):
        """
        wrapper to get value of a given model key from models dict
        """
        value = self.models.get(key)
        return value

    def _set_k(self):
        """
        Helper function, get the penalty coefficient for AIC/BIC calculation.
        """
        if self.score_name == "aic":
            self.k = 2
        elif self.score_name == "bic":
            self.k = np.log(self.n_count)
        else:
            raise ValueError("Wrong score name given: {}. Only 'aic' or 'bic' acceptable.".format(self.score_name))

    @staticmethod
    def get_dfe(model, str_mask):
        dfe = sum(HeteroStepwise.string2mask(str_mask))
        if model.fit_intercept:
            dfe += 1
            LOGGER.debug("fit_intercept detected, 1 is added to dfe")
        return dfe

    def get_step_best(self, step_models):
        best_score = None
        best_model = ""
        for model in step_models:
            model_info = self.models_trained[model]
            score = model_info.get_score()
            if score is None:
                continue
            if best_score is None or score < best_score:
                best_score = score
                best_model = model
        LOGGER.info(f"step {self.n_step}, best model {best_model}, best_score {best_score}")
        return best_model

    @staticmethod
    def drop_one(mask_to_drop):
        for i in np.nonzero(mask_to_drop)[0]:
            new_mask = np.copy(mask_to_drop)
            new_mask[i] = 0
            if sum(new_mask) > 0:
                yield new_mask

    @staticmethod
    def add_one(mask_to_add):
        for i in np.where(mask_to_add < 1)[0]:
            new_mask = np.copy(mask_to_add)
            new_mask[i] = 1
            yield new_mask

    def check_stop(self, new_host_mask, new_guest_mask, host_mask, guest_mask):
        # initial step
        if self.n_step == 0:
            return False
        # if model not updated
        if np.array_equal(new_host_mask, host_mask) and np.array_equal(new_guest_mask, guest_mask):
            LOGGER.debug("masks not changed, check_stop returns True")
            return True
        # if full model is the best
        if sum(new_host_mask < 1) == 0 and sum(new_guest_mask < 1) == 0 and self.n_step > 0:
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
        if self.n_step >= self.max_step:
            LOGGER.debug("max step reached, check_stop returns True")
            return True
        return False

    def get_intercept_loss(self, model, data):
        y = np.array([x[1] for x in data.mapValues(lambda v: v.label).collect()])
        X = np.ones((len(y), 1))
        if model.model_name == 'HeteroLinearRegression' or model.model_name == 'HeteroPoissonRegression':
            intercept_model = LinearRegression(fit_intercept=False)
            trained_model = intercept_model.fit(X, y)
            pred = trained_model.predict(X)
            loss = metrics.mean_squared_error(y, pred) / 2
        elif model.model_name == 'HeteroLogisticRegression':
            intercept_model = LogisticRegression(penalty='l1', C=1e8, fit_intercept=False, solver='liblinear')
            trained_model = intercept_model.fit(X, y)
            pred = trained_model.predict(X)
            loss = metrics.log_loss(y, pred)
        else:
            raise ValueError("Unknown model received. Stepwise stopped.")
        self.intercept = intercept_model.intercept_
        return loss

    def get_ic_val(self, model, model_mask):
        if self.role != consts.ARBITER:
            return None, None
        if len(model.loss_history) == 0:
            raise ValueError("Arbiter has no loss history. Stepwise does not support model without total loss.")
        # get final loss from loss history for criteria calculation
        loss = model.loss_history[-1]
        dfe = HeteroStepwise.get_dfe(model, model_mask)
        ic_val = self.IC_computer.compute(self.k, self.n_count, dfe, loss)
        if np.isinf(ic_val):
            raise ValueError("Loss value of infinity obtained. Stepwise stopped.")
        return loss, ic_val

    def get_ic_val_guest(self, model, train_data):
        if not model.fit_intercept:
            return None, None
        loss = self.get_intercept_loss(model, train_data)
        # intercept only model has dfe = 1
        dfe = 1
        ic_val = self.IC_computer.compute(self.k, self.n_count, dfe, loss)
        return loss, ic_val

    def _run_step(self, model, train_data, validate_data, feature_mask, n_model, model_mask):
        if self.direction == 'forward' and self.n_step == 0:
            if self.role == consts.GUEST:
                loss, ic_val = self.get_ic_val_guest(model, train_data)
                LOGGER.info("step {} n_model {}: ic_val {}".format(self.n_step, n_model, ic_val))
                model_info = ModelInfo(self.n_step, n_model, ic_val, loss, self.step_direction)
                self.models_trained[model_mask] = model_info
                model_key = model_info.get_key()
                self._put_model(model_key, model)
            else:
                model_info = ModelInfo(self.n_step, n_model, None, None, self.step_direction)
                self.models_trained[model_mask] = model_info
                model_key = model_info.get_key()
                self._put_model(model_key, model)
            return
        curr_step = Step()
        curr_step.set_step_info((self.n_step, n_model))
        trained_model = curr_step.run(model, train_data, validate_data, feature_mask)
        loss, ic_val = self.get_ic_val(trained_model, model_mask)
        LOGGER.info("step {} n_model {}: ic_val {}".format(self.n_step, n_model, ic_val))
        model_info = ModelInfo(self.n_step, n_model, ic_val, loss, self.step_direction)
        self.models_trained[model_mask] = model_info
        model_key = model_info.get_key()
        self._put_model(model_key, trained_model)

    def sync_data_info(self, data):
        if self.role == consts.ARBITER:
            return self.arbiter_sync_data_info()
        else:
            return self.client_sync_data_info(data)

    def arbiter_sync_data_info(self):
        n_host, j_host = self.transfer_variable.host_data_info.get(idx=0)
        n_guest, j_guest = self.transfer_variable.guest_data_info.get(idx=0)
        self.n_count = n_host
        return j_host, j_guest

    def client_sync_data_info(self, data):
        n, j = data.count(), data_overview.get_features_shape(data)
        self.n_count = n
        if self.role == consts.HOST:
            self.transfer_variable.host_data_info.remote((n, j), role=consts.ARBITER, idx=0)
            self.transfer_variable.host_data_info.remote((n, j), role=consts.GUEST, idx=0)
            j_host = j
            n_guest, j_guest = self.transfer_variable.guest_data_info.get(idx=0)
        else:
            self.transfer_variable.guest_data_info.remote((n, j), role=consts.ARBITER, idx=0)
            self.transfer_variable.guest_data_info.remote((n, j), role=consts.HOST, idx=0)
            j_guest = j
            n_host, j_host = self.transfer_variable.host_data_info.get(idx=0)
        return j_host, j_guest

    def get_to_enter(self, host_mask, guest_mask, all_features):
        if self.role == consts.GUEST:
            to_enter = [all_features[i] for i in np.where(guest_mask < 1)[0]]

        elif self.role == consts.HOST:
            to_enter = [all_features[i] for i in np.where(host_mask < 1)[0]]
        else:
            to_enter = []
        return to_enter

    def record_step_best(self, step_best, host_mask, guest_mask, data_instances, model):
        metas = {"host_mask": host_mask.tolist(), "guest_mask": guest_mask.tolist(),
                 "score_name": self.score_name}
        metas["number_in"] = int(sum(host_mask) + sum(guest_mask))
        metas["direction"] = self.direction
        metas["n_count"] = int(self.n_count)

        model_info = self.models_trained[step_best]
        loss = model_info.get_loss()
        ic_val = model_info.get_score()
        metas["loss"] = loss
        metas["current_ic_val"] = ic_val
        metas["fit_intercept"] = model.fit_intercept

        model_key = model_info.get_key()
        model_dict = self._get_model(model_key)

        if self.role != consts.ARBITER:
            all_features = data_instances.schema.get('header')
            metas["all_features"] = all_features
            metas["to_enter"] = self.get_to_enter(host_mask, guest_mask, all_features)
            model_param = list(model_dict.get('model').values())[0].get(
                model.model_param_name)
            param_dict = MessageToDict(model_param)
            metas["intercept"] = param_dict.get("intercept", None)
            metas["weight"] = param_dict.get("weight", {})
            metas["header"] = param_dict.get("header", [])
            if self.n_step == 0 and self.direction == "forward":
                metas["intercept"] = self.intercept

        metric_name = f"stepwise_{self.n_step}"
        metric = [Metric(metric_name, float(self.n_step))]
        model.callback_metric(metric_name=metric_name, metric_namespace=self.metric_namespace, metric_data=metric)
        model.tracker.set_metric_meta(metric_name=metric_name, metric_namespace=self.metric_namespace,
                                      metric_meta=MetricMeta(name=metric_name, metric_type=self.metric_type,
                                                             extra_metas=metas))
        LOGGER.info(f"metric_name: {metric_name}, metas: {metas}")
        return

    def sync_step_best(self, step_models):
        if self.role == consts.ARBITER:
            step_best = self.get_step_best(step_models)
            self.transfer_variable.step_best.remote(step_best, role=consts.HOST, suffix=(self.n_step,))
            self.transfer_variable.step_best.remote(step_best, role=consts.GUEST, suffix=(self.n_step,))
            LOGGER.info(f"step {self.n_step}, step_best sent is {step_best}")
        else:
            step_best = self.transfer_variable.step_best.get(suffix=(self.n_step,))[0]
            LOGGER.info(f"step {self.n_step}, step_best received is {step_best}")
        return step_best

    @staticmethod
    def mask2string(host_mask, guest_mask):
        mask = np.append(host_mask, guest_mask)
        string_repr = ''.join('1' if i else '0' for i in mask)
        return string_repr

    @staticmethod
    def string2mask(string_repr):
        mask = np.fromiter(map(int, string_repr), dtype=bool)
        return mask

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

    def get_IC_computer(self, model):
        if model.model_name == 'HeteroLinearRegression':
            return IC_Approx()
        else:
            return IC()

    def run(self, component_parameters, train_data, validate_data, model):
        LOGGER.info("Enter stepwise")
        self._init_model(component_parameters)
        j_host, j_guest = self.sync_data_info(train_data)
        if self.backward:
            host_mask, guest_mask = np.ones(j_host, dtype=bool), np.ones(j_guest, dtype=bool)
        else:
            host_mask, guest_mask = np.zeros(j_host, dtype=bool), np.zeros(j_guest, dtype=bool)
        self.IC_computer = self.get_IC_computer(model)
        self._set_k()
        while self.n_step <= self.max_step:
            LOGGER.info("Enter step {}".format(self.n_step))
            step_models = set()
            step_models.add(HeteroStepwise.mask2string(host_mask, guest_mask))
            n_model = 0
            if self.backward:
                self.step_direction = "backward"
                LOGGER.info("step {}, direction: {}".format(self.n_step, self.step_direction))
                if self.n_step == 0:
                    backward_gen = [[host_mask, guest_mask]]
                else:
                    backward_host, backward_guest = HeteroStepwise.drop_one(host_mask), HeteroStepwise.drop_one(
                        guest_mask)
                    backward_gen = itertools.chain(zip(backward_host, itertools.cycle([guest_mask])),
                                                   zip(itertools.cycle([host_mask]), backward_guest))
                for curr_host_mask, curr_guest_mask in backward_gen:
                    model_mask = HeteroStepwise.mask2string(curr_host_mask, curr_guest_mask)
                    step_models.add(model_mask)
                    if model_mask not in self.models_trained:
                        if self.role == consts.ARBITER:
                            feature_mask = None
                        elif self.role == consts.HOST:
                            feature_mask = curr_host_mask
                        else:
                            feature_mask = curr_guest_mask
                        self._run_step(model, train_data, validate_data, feature_mask, n_model, model_mask)
                        n_model += 1

            if self.forward:
                self.step_direction = "forward"
                LOGGER.info("step {}, direction: {}".format(self.n_step, self.step_direction))
                forward_host, forward_guest = HeteroStepwise.add_one(host_mask), HeteroStepwise.add_one(guest_mask)
                if sum(guest_mask) + sum(host_mask) == 0:
                    if self.n_step == 0:
                        forward_gen = [[host_mask, guest_mask]]
                    else:
                        forward_gen = itertools.product(list(forward_host), list(forward_guest))
                else:
                    forward_gen = itertools.chain(zip(forward_host, itertools.cycle([guest_mask])),
                                                  zip(itertools.cycle([host_mask]), forward_guest))
                for curr_host_mask, curr_guest_mask in forward_gen:
                    model_mask = HeteroStepwise.mask2string(curr_host_mask, curr_guest_mask)
                    step_models.add(model_mask)
                    LOGGER.info(f"step {self.n_step}, mask {model_mask}")
                    if model_mask not in self.models_trained:
                        if self.role == consts.ARBITER:
                            feature_mask = None
                        elif self.role == consts.HOST:
                            feature_mask = curr_host_mask
                        else:
                            feature_mask = curr_guest_mask
                        self._run_step(model, train_data, validate_data, feature_mask, n_model, model_mask)
                        n_model += 1
            # forward step 0
            if sum(host_mask) + sum(guest_mask) == 0 and self.n_step == 0:
                model_mask = HeteroStepwise.mask2string(host_mask, guest_mask)
                self.record_step_best(model_mask, host_mask, guest_mask, train_data, model)
                self.n_step += 1
                continue
            old_host_mask, old_guest_mask = host_mask, guest_mask
            step_best = self.sync_step_best(step_models)
            step_best_mask = HeteroStepwise.string2mask(step_best)
            host_mask, guest_mask = step_best_mask[:j_host], step_best_mask[j_host:]
            LOGGER.debug("step {}, best_host_mask {}, best_guest_mask {}".format(self.n_step, host_mask, guest_mask))
            self.stop_stepwise = self.check_stop(host_mask, guest_mask, old_host_mask, old_guest_mask)
            if self.stop_stepwise:
                break
            self.record_step_best(step_best, host_mask, guest_mask, train_data, model)
            self.n_step += 1

        mask_string = HeteroStepwise.mask2string(host_mask, guest_mask)
        model_info = self.models_trained[mask_string]
        best_model_key = model_info.get_key()
        best_model = self._get_model(best_model_key)
        model.load_model(best_model)
