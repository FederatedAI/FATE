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

from federatedml.loss.cross_entropy import SigmoidBinaryCrossEntropyLoss
from federatedml.loss.cross_entropy import SoftmaxCrossEntropyLoss
from federatedml.loss.regression_loss import LeastSquaredErrorLoss
from federatedml.loss.regression_loss import LeastAbsoluteErrorLoss
from federatedml.loss.regression_loss import HuberLoss
from federatedml.loss.regression_loss import FairLoss
from federatedml.loss.regression_loss import LogCoshLoss
from federatedml.loss.regression_loss import TweedieLoss

__all__ = ["SigmoidBinaryCrossEntropyLoss",
           "SoftmaxCrossEntropyLoss",
           "LeastSquaredEroorLoss",
           "LeastAbsoluteErrorLoss",
           "HuberLoss",
           "FairLoss",
           "LogCoshLoss",
           "TweedieLoss"]
