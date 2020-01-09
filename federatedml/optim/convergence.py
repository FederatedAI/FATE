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

from arch.api.utils import log_utils
from federatedml.util import fate_operator

LOGGER = log_utils.getLogger()


class _ConvergeFunction:
    def __init__(self, eps):
        self.eps = eps

    def is_converge(self, loss): pass


class _DiffConverge(_ConvergeFunction):
    """
    Judge convergence by the difference between two iterations.
    If the difference is smaller than eps, converge flag will be provided.
    """

    def __init__(self, eps):
        super().__init__(eps=eps)
        self.pre_loss = None

    def is_converge(self, loss):
        LOGGER.debug("In diff converge function, pre_loss: {}, current_loss: {}".format(self.pre_loss, loss))

        converge_flag = False
        if self.pre_loss is None:
            pass
        elif abs(self.pre_loss - loss) < self.eps:
            converge_flag = True
        self.pre_loss = loss
        return converge_flag


class _AbsConverge(_ConvergeFunction):
    """
    Judge converge by absolute loss value. When loss value smaller than eps, converge flag
    will be provided.
    """

    def is_converge(self, loss):
        if loss <= self.eps:
            converge_flag = True
        else:
            converge_flag = False
        return converge_flag


class _WeightDiffConverge(_ConvergeFunction):
    """
    Use 2-norm of weight difference to judge whether converge or not.
    """

    def __init__(self, eps):
        super().__init__(eps=eps)
        self.pre_weight = None

    def is_converge(self, weight):
        if self.pre_weight is None:
            self.pre_weight = weight
            return False

        weight_diff = fate_operator.norm(self.pre_weight - weight)
        self.pre_weight = weight
        if weight_diff < self.eps * np.max([fate_operator.norm(weight), 1]):
            return True
        return False


def converge_func_factory(early_stop, tol):
    # try:
    #     converge_func = param.converge_func
    #     eps = param.eps
    # except AttributeError:
    #     raise AttributeError("Converge Function parameters has not been totally set")

    if early_stop == 'diff':
        return _DiffConverge(tol)
    elif early_stop == 'weight_diff':
        return _WeightDiffConverge(tol)
    elif early_stop == 'abs':
        return _AbsConverge(tol)
    else:
        raise NotImplementedError("Converge Function method cannot be recognized: {}".format(early_stop))
