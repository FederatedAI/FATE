#
#  Copyright 2023 The FATE Authors. All Rights Reserved.
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

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


class LRScheduler:
    def __init__(self, method=None, lr_params=None, iters=0):
        self.method = method
        self.lr_params = lr_params
        self.iters = iters
        self.lr_scheduler = None

    def init_scheduler(self, optimizer):
        self.lr_scheduler = lr_scheduler_factory(optimizer, self.method, self.lr_params)

    def step(self):
        self.lr_scheduler.step()
        self.iters += 1

    @property
    def lr(self):
        return self.lr_scheduler.get_last_lr()[0]

    def state_dict(self):
        return {"lr_scheduler": self.lr_scheduler.state_dict(), "method": self.method, "lr_params": self.lr_params}

    def load_state_dict(self, dict, optimizer):
        self.method = dict["method"]
        self.lr_params = dict["lr_params"]
        self.init_scheduler(optimizer)
        self.lr_scheduler.load_state_dict(dict["lr_scheduler"])

    def get_last_lr(self):
        return self.get_last_lr()


class Optimizer(object):
    def __init__(self, method=None, penalty=None, alpha=None, optim_param: dict = None, iters: int = 0):
        self.method = method
        self.optim_param = optim_param
        self.iters = iters
        self.l2_penalty = True if penalty == "l2" else False
        self.l1_penalty = True if penalty == "l1" else False
        self.alpha = alpha

        self.model_parameter = None
        self.prev_model_parameter = None
        self.optimizer = None

    def init_optimizer(self, model_parameter_length=None, model_parameter=None, dtype=torch.float32):
        # allow group of parameter in future
        if model_parameter_length is not None:
            model_parameter = torch.nn.parameter.Parameter(
                torch.zeros((model_parameter_length, 1), requires_grad=True, dtype=dtype)
            )
        self.model_parameter = model_parameter
        self.optimizer = optimizer_factory([model_parameter], self.method, self.optim_param)
        # for regularization
        # self.alpha = self.optimizer.state_dict()['param_groups'][0]['alpha']

    def step(self, gradient):
        # logger.info(f"before copy, model parameter: {self.model_parameter}")
        self.prev_model_parameter = self.model_parameter.data.clone()
        self.model_parameter.grad = gradient
        self.optimizer.step()
        # logger.info(f"after step, model parameter: {self.model_parameter}")

    def get_delta_gradients(self):
        # logger.info(f"gradient: {self.model_parameter.grad}, prev model parameter: {self.prev_model_parameter},"
        # f"delta grad: {self.prev_model_parameter.data - self.model_parameter.data}")
        if self.prev_model_parameter is not None:
            return self.prev_model_parameter.data - self.model_parameter.data
        else:
            raise ValueError(f"No optimization history found, please check.")

    def shrinkage_val(self, lr):
        this_step_size = lr / np.sqrt(self.iters)
        return self.alpha * this_step_size

    def state_dict(self):
        optimizer_state_dict = self.optimizer.state_dict()
        state_all = optimizer_state_dict["state"].get(0, {})
        for k, v in state_all.items():
            if isinstance(v, torch.Tensor):
                state_all[k] = v.tolist()
        dtype = str(self.model_parameter.dtype).split(".", -1)[-1]
        return {
            "l2_penalty": self.l2_penalty,
            "l1_penalty": self.l1_penalty,
            "alpha": self.alpha,
            "optimizer": optimizer_state_dict,
            "method": self.method,
            "optim_param": self.optim_param,
            "model_parameter": self.model_parameter.tolist(),
            "model_parameter_dtype": dtype,
        }

    def load_state_dict(self, state_dict):
        self.l2_penalty = state_dict["l2_penalty"]
        self.l1_penalty = state_dict["l1_penalty"]
        self.alpha = state_dict["alpha"]
        self.method = state_dict["method"]
        self.optim_param = state_dict["optim_param"]
        dtype = state_dict["model_parameter_dtype"]
        self.init_optimizer(
            model_parameter=torch.nn.parameter.Parameter(
                torch.tensor(state_dict["model_parameter"], dtype=getattr(torch, dtype))
            )
        )
        state = state_dict["optimizer"]
        state_all = state["state"].get(0, {})
        for k, v in state_all.items():
            if isinstance(v, list):
                state_all[k] = torch.tensor(v)
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def set_iters(self, new_iters):
        self.iters = new_iters

    def _l1_updator(self, model_weights, gradient, fit_intercept, lr):
        if fit_intercept:
            gradient_without_intercept = gradient[:-1]
            coef_ = model_weights[:-1]
        else:
            gradient_without_intercept = gradient
            coef_ = model_weights

        new_weights = torch.sign(coef_ - gradient_without_intercept) * torch.maximum(
            torch.tensor([0]), torch.abs(coef_ - gradient_without_intercept) - self.shrinkage_val(lr)
        )

        if fit_intercept:
            new_intercept = model_weights[-1] - gradient[-1]
            new_weights = torch.concat((new_weights, new_intercept.reshape((1, 1))))

        return new_weights

    def add_regular_to_grad(self, grad, model_weights, fit_intercept=False):
        if self.l2_penalty:
            if fit_intercept:
                weights_sum = torch.concat((model_weights[:-1], torch.tensor([[0]])))
                # logger.info(f"grad: {grad}, weights sum: {weights_sum}")
                new_grad = grad + self.alpha * weights_sum
            else:
                new_grad = grad + self.alpha * model_weights
        else:
            new_grad = grad

        return new_grad

    def regularization_update(self, model_weights, grad, fit_intercept, lr, prev_round_weights=None):
        if self.l1_penalty:
            model_weights = self._l1_updator(model_weights, grad, fit_intercept, lr)
        else:
            model_weights = model_weights - grad
        """elif self.l2_penalty:
                    model_weights = self._l2_updator(model_weights, grad)
                """
        """if prev_round_weights is not None:  # additional proximal term for homo
            coef_ = model_weights.unboxed

            if model_weights.fit_intercept:
                coef_without_intercept = coef_[: -1]
            else:
                coef_without_intercept = coef_

            coef_without_intercept -= self.mu * (model_weights.coef_ - prev_round_weights.coef_)

            if model_weights.fit_intercept:
                new_coef_ = np.append(coef_without_intercept, coef_[-1])
            else:
                new_coef_ = coef_without_intercept

            model_weights = LinearModelWeights(new_coef_,
                                               model_weights.fit_intercept,
                                               model_weights.raise_overflow_error)"""
        return model_weights

    def __l1_loss_norm(self, model_weights):
        loss_norm = torch.sum(self.alpha * model_weights)
        return loss_norm.reshape((1, 1))

    def __l2_loss_norm(self, model_weights):
        loss_norm = 0.5 * self.alpha * torch.matmul(model_weights.T, model_weights)
        return loss_norm

    """def __add_proximal(self, model_weights, prev_round_weights):
        prev_round_coef_ = prev_round_weights.coef_
        coef_ = model_weights.coef_
        diff = coef_ - prev_round_coef_
        loss_norm = self.mu * 0.5 * np.dot(diff, diff)
        return loss_norm
    """

    def loss_norm(self, model_weights, prev_round_weights=None):
        """
        proximal_term = None
        if prev_round_weights is not None:
            proximal_term = self.__add_proximal(model_weights, prev_round_weights)
        """

        if self.l1_penalty:
            loss_norm_value = self.__l1_loss_norm(model_weights)
        elif self.l2_penalty:
            loss_norm_value = self.__l2_loss_norm(model_weights)
        else:
            loss_norm_value = None

        """# additional proximal term
        if loss_norm_value is None:
            loss_norm_value = proximal_term
        elif proximal_term is not None:
            loss_norm_value += proximal_term"""
        return loss_norm_value

    """def hess_vector_norm(self, delta_s: LinearModelWeights):
        if self.penalty == consts.L1_PENALTY:
            return LinearModelWeights(np.zeros_like(delta_s.unboxed),
                                      fit_intercept=delta_s.fit_intercept,
                                      raise_overflow_error=delta_s.raise_overflow_error)
        elif self.penalty == consts.L2_PENALTY:
            return LinearModelWeights(self.alpha * np.array(delta_s.unboxed),
                                      fit_intercept=delta_s.fit_intercept,
                                      raise_overflow_error=delta_s.raise_overflow_error)
        else:
            return LinearModelWeights(np.zeros_like(delta_s.unboxed),
                                      fit_intercept=delta_s.fit_intercept,
                                      raise_overflow_error=delta_s.raise_overflow_error)
    """

    def update_weights(self, model_weights, grad, fit_intercept, lr, prev_round_weights=None, has_applied=True):
        """if not has_applied:
            grad = self.add_regular_to_grad(grad, model_weights)
            delta_grad = self.apply_gradients(grad)
        else:"""
        # logger.info(
        #     f"before update, model weights: {model_weights}, delta_grad: {grad}")
        delta_grad = grad
        model_weights = self.regularization_update(model_weights, delta_grad, fit_intercept, lr, prev_round_weights)
        # (f"after update, model weights: {model_weights}")

        return model_weights


def separate(value, size_list):
    """
    Separate value in order to several set according size_list
    Parameters
    ----------
    value: 2d-tensor, input data
    size_list: list, each set size
    Returns
    ----------
    list
        separated 2d-tensors of sizes given in size_list
    """
    separate_res = []
    cur = 0
    for size in size_list:
        separate_res.append(value[cur : cur + size, :])
        cur += size
    return separate_res


def optimizer_factory(model_parameter, optimizer_type, optim_params):
    optimizer_params = optim_params

    if optimizer_type == "adadelta":
        return torch.optim.Adadelta(model_parameter, **optimizer_params)
    elif optimizer_type == "adagrad":
        return torch.optim.Adagrad(model_parameter, **optimizer_params)
    elif optimizer_type == "adam":
        return torch.optim.Adam(model_parameter, **optimizer_params)
    elif optimizer_type == "adamw":
        return torch.optim.AdamW(model_parameter, **optimizer_params)
    elif optimizer_type == "adamax":
        return torch.optim.Adamax(model_parameter, **optimizer_params)
    elif optimizer_type == "asgd":
        return torch.optim.ASGD(model_parameter, **optimizer_params)
    elif optimizer_type == "nadam":
        return torch.optim.NAdam(model_parameter, **optimizer_params)
    elif optimizer_type == "radam":
        return torch.optim.RAdam(model_parameter, **optimizer_params)
    elif optimizer_type == "rmsprop":
        return torch.optim.RMSprop(model_parameter, **optimizer_params)
    elif optimizer_type == "rprop":
        return torch.optim.Rprop(model_parameter, **optimizer_params)
    elif optimizer_type == "sgd":
        return torch.optim.SGD(model_parameter, **optimizer_params)
    else:
        raise NotImplementedError("Optimize method cannot be recognized: {}".format(optimizer_type))


def lr_scheduler_factory(optimizer, method, scheduler_param):
    scheduler_method = method
    if scheduler_method == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, **scheduler_param)
    elif scheduler_method == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_param)
    elif scheduler_method == "linear":
        return torch.optim.lr_scheduler.LinearLR(optimizer, **scheduler_param)
    else:
        raise NotImplementedError(f"Learning rate method cannot be recognized: {scheduler_method}")
