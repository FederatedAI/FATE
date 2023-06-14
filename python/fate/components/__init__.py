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
from typing import Dict, Literal, Optional

T_ROLE = Literal["guest", "host", "arbiter"]
T_STAGE = Literal["train", "predict", "default"]
T_LABEL = Literal["trainable"]


class Role:
    def __init__(self, name: T_ROLE) -> None:
        self.name: T_ROLE = name

    @property
    def is_guest(self) -> bool:
        return self.name == "guest"

    @property
    def is_host(self) -> bool:
        return self.name == "host"

    @property
    def is_arbiter(self) -> bool:
        return self.name == "arbiter"


GUEST = Role("guest")
HOST = Role("host")
ARBITER = Role("arbiter")

T_ROLE = Literal["guest", "host", "arbiter"]
T_STAGE = Literal["train", "predict", "default"]
T_LABEL = Literal["trainable"]


class Stage:
    def __init__(self, name: str) -> None:
        self.name = name

    @property
    def is_train(self):
        return self.name == TRAIN.name

    @property
    def is_predict(self):
        return self.name == PREDICT.name

    @property
    def is_train_eval(self):
        return self.name == CROSS_VALIDATION.name

    @property
    def is_default(self):
        return self.name == DEFAULT.name


TRAIN = Stage("train")
PREDICT = Stage("predict")
CROSS_VALIDATION = Stage("cross_validation")
DEFAULT = Stage("default")


class LABELS:
    TRAINABLE = "trainable"
