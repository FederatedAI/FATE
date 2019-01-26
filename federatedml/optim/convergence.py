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

from federatedml.util import consts


class ConvergeFunction:

    def __init__(self, eps):
        self.eps = eps

    def is_converge(self, loss): pass


class DiffConverge(ConvergeFunction):
    """
    Judge convergence by the difference between two iterations.
    If the difference is smaller than eps, converge flag will be provided.
    """

    def __init__(self, pre_loss=None, eps=consts.FLOAT_ZERO):
        super(DiffConverge, self).__init__(eps=eps)
        self.pre_loss = pre_loss

    def is_converge(self, loss):
        converge_flag = False
        if self.pre_loss is None:
            pass
        elif abs(self.pre_loss - loss) < self.eps:
            converge_flag = True
        self.pre_loss = loss
        return converge_flag


class AbsConverge(ConvergeFunction):
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
