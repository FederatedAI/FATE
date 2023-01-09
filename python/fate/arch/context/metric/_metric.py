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
from ._type import Metric


class ScalarMetric(Metric):
    type = "scalar"

    def __init__(self, scalar) -> None:
        self.scalar = scalar

    def dict(self):
        return self.scalar


class LossMetric(Metric):
    type = "loss"

    def __init__(self, loss) -> None:
        self.loss = loss

    def dict(self) -> dict:
        return self.loss


class AccuracyMetric(Metric):
    type = "accuracy"

    def __init__(self, accuracy) -> None:
        self.accuracy = accuracy

    def dict(self) -> dict:
        return self.accuracy


class AUCMetric(Metric):
    type = "auc"

    def __init__(self, auc) -> None:
        self.auc = auc

    def dict(self) -> dict:
        return self.auc
