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

    def is_cross_validation(self):
        return self.name == CROSS_VALIDATION.name

    @classmethod
    def from_str(cls, stage: str):
        if stage == "train":
            return TRAIN
        elif stage == "predict":
            return PREDICT
        elif stage == "cross_validation":
            return CROSS_VALIDATION
        elif stage == "default":
            return DEFAULT
        else:
            raise ValueError(f"stage {stage} is not supported")

    def __str__(self):
        return f"Stage<{self.name}>"

    def __repr__(self):
        return f"Stage<{self.name}>"


TRAIN = Stage("train")
PREDICT = Stage("predict")
CROSS_VALIDATION = Stage("cross_validation")
DEFAULT = Stage("default")
