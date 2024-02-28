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
from fate.components.components.nn.component_utils import (
    prepare_runner_class,
    prepare_context_and_role,
    train_procedure,
    predict_procedure,
)
from fate.components.components.utils import consts
from fate.components.core import ARBITER, GUEST, HOST, Role, cpn


logger = logging.getLogger(__name__)


def is_path(s):
    return os.path.exists(s)


""""
Output functions
"""


@cpn.component(roles=[GUEST, HOST, ARBITER])
def homo_nn(ctx, role):
    ...


@homo_nn.train()
def train(
    ctx: Context,
    role: Role,
    train_data: cpn.dataframe_input(roles=[GUEST, HOST]) | cpn.data_directory_input(),
    validate_data: cpn.dataframe_input(roles=[GUEST, HOST], optional=True) | cpn.data_directory_input(optional=True),
    runner_module: cpn.parameter(type=str, default="homo_default_runner", desc="name of your runner script"),
    runner_class: cpn.parameter(type=str, default="DefaultRunner", desc="class name of your runner class"),
    runner_conf: cpn.parameter(type=dict, default={}, desc="the parameter dict of the NN runner class"),
    source: cpn.parameter(type=str, default=None, desc="path to your runner script folder"),
    train_output_data: cpn.dataframe_output(roles=[GUEST, HOST], optional=True),
    output_model: cpn.model_directory_output(roles=[GUEST, HOST], optional=True),
    warm_start_model: cpn.model_directory_input(roles=[GUEST, HOST], optional=True),
):
    if role.is_guest or role.is_host:  # is client
        train_procedure(
            ctx,
            role,
            train_data,
            validate_data,
            runner_module,
            runner_class,
            runner_conf,
            source,
            train_output_data,
            output_model,
            warm_start_model,
        )

    elif role.is_arbiter:  # is server
        runner: NNRunner = prepare_runner_class(runner_module, runner_class, runner_conf, source)
        prepare_context_and_role(runner, ctx, role, consts.TRAIN)
        runner.train()


@homo_nn.predict()
def predict(
    ctx,
    role: Role,
    test_data: cpn.dataframe_input(roles=[GUEST, HOST]) | cpn.data_directory_input(),
    input_model: cpn.model_directory_input(roles=[GUEST, HOST]),
    test_output_data: cpn.dataframe_output(roles=[GUEST, HOST], optional=True),
):
    if role.is_guest or role.is_host:  # is client
        predict_procedure(ctx, role, test_data, input_model, test_output_data)

    elif role.is_arbiter:  # is server
        logger.info("arbiter skip predict")
