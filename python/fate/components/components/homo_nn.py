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
from fate.components import (
    ARBITER,
    GUEST,
    HOST,
    DatasetArtifact,
    Input,
    LossMetrics,
    ModelArtifact,
    Output,
    Role,
    cpn,
    params,
)
from fate.components.components.nn.setup.fate_setup import FateSetup


@cpn.component(roles=[GUEST, HOST, ARBITER])
def homo_nn(ctx, role):
    ...


@homo_nn.train
@cpn.artifact("train_data", type=Input[DatasetArtifact], roles=[GUEST, HOST], desc="training data")
@cpn.artifact("validate_data", type=Input[DatasetArtifact], optional=True, roles=[GUEST, HOST], desc="validation data")
@cpn.parameter("learning_rate", type=params.learning_rate_param(), default=0.1, desc="learning rate")
@cpn.artifact("train_output_data", type=Output[DatasetArtifact], roles=[GUEST, HOST])
@cpn.artifact("train_output_metric", type=Output[LossMetrics], roles=[ARBITER])
@cpn.artifact("output_model", type=Output[ModelArtifact], roles=[GUEST, HOST])
def train(
    ctx,
    role: Role,
    train_data,
    validate_data,
    learning_rate,
    max_iter,
    batch_size,
    train_output_data,
    train_output_metric,
    output_model,
):
   print('running homo nn cwj 114514')