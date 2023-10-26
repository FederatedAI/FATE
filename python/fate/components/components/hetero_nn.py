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
import logging
import os
from fate.arch import Context
from fate.components.components.nn.nn_runner import NNRunner
from fate.components.components.utils import consts
from fate.components.core import GUEST, HOST, Role, cpn
from fate.arch.dataframe import DataFrame
from fate.components.components.utils.tools import add_dataset_type
from fate.components.components.nn.component_utils import train_guest_and_host


logger = logging.getLogger(__name__)


@cpn.component(roles=[GUEST, HOST])
def hetero_nn(ctx, role):
    ...


@hetero_nn.train()
def train(
    ctx: Context,
    role: Role,
    train_data: cpn.dataframe_input(roles=[GUEST, HOST]),
    validate_data: cpn.dataframe_input(roles=[GUEST, HOST], optional=True),
    runner_module: cpn.parameter(type=str, default="hetero_default_runner", desc="name of your runner script"),
    runner_class: cpn.parameter(type=str, default="DefaultRunner", desc="class name of your runner class"),
    runner_conf: cpn.parameter(type=dict, default={}, desc="the parameter dict of the NN runner class"),
    source: cpn.parameter(type=str, default=None, desc="path to your runner script folder"),
    train_data_output: cpn.dataframe_output(roles=[GUEST, HOST], optional=True),
    train_model_output: cpn.model_directory_output(roles=[GUEST, HOST], optional=True),
    train_model_input: cpn.model_directory_input(roles=[GUEST, HOST], optional=True)
):

    train_guest_and_host(
        ctx,
        role,
        train_data,
        validate_data,
        runner_module,
        runner_class,
        runner_conf,
        source,
        train_data_output,
        train_model_output,
        train_model_input
    )


@hetero_nn.predict()
def predict(
        ctx: Context,
        role: Role, test_data: cpn.dataframe_input(roles=[GUEST, HOST]),
        predict_model_input: cpn.model_directory_input(roles=[GUEST, HOST]),
        predict_data_output: cpn.dataframe_output(roles=[GUEST, HOST], optional=True)
):
    pass

