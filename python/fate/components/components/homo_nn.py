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
from fate.components.components.nn.component_utils import (prepare_runner_class, prepare_context_and_role,
                                                           get_input_data, train_guest_and_host)
from fate.components.components.utils import consts
from fate.components.core import ARBITER, GUEST, HOST, Role, cpn
from fate.arch.dataframe import DataFrame
from fate.components.components.utils.tools import add_dataset_type


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
    train_data: cpn.dataframe_input(roles=[GUEST, HOST]),
    validate_data: cpn.dataframe_input(roles=[GUEST, HOST], optional=True),
    runner_module: cpn.parameter(type=str, default="homo_default_runner", desc="name of your runner script"),
    runner_class: cpn.parameter(type=str, default="DefaultRunner", desc="class name of your runner class"),
    runner_conf: cpn.parameter(type=dict, default={}, desc="the parameter dict of the NN runner class"),
    source: cpn.parameter(type=str, default=None, desc="path to your runner script folder"),
    train_data_output: cpn.dataframe_output(roles=[GUEST, HOST], optional=True),
    train_model_output: cpn.model_directory_output(roles=[GUEST, HOST], optional=True),
    train_model_input: cpn.model_directory_input(roles=[GUEST, HOST], optional=True)
):


    if role.is_guest or role.is_host:  # is client

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

    elif role.is_arbiter:  # is server
        runner: NNRunner = prepare_runner_class(
            runner_module, runner_class, runner_conf, source)
        prepare_context_and_role(runner, ctx, role, consts.TRAIN)
        runner.train()


@homo_nn.predict()
def predict(
    ctx, role: Role, test_data: cpn.dataframe_input(
        roles=[
            GUEST, HOST]), predict_model_input: cpn.model_directory_input(
                roles=[
                    GUEST, HOST]), predict_data_output: cpn.dataframe_output(
                        roles=[
                            GUEST, HOST], optional=True)):

    if role.is_guest or role.is_host:  # is client

        model_conf = predict_model_input.get_metadata()
        runner_module = model_conf['runner_module']
        runner_class = model_conf['runner_class']
        runner_conf = model_conf['runner_conf']
        source = model_conf['source']
        saved_model_path = str(predict_model_input.get_directory())
        test_data_ = get_input_data(consts.PREDICT, test_data)
        runner: NNRunner = prepare_runner_class(
            runner_module, runner_class, runner_conf, source)
        prepare_context_and_role(runner, ctx, role, consts.PREDICT)
        test_pred = runner.predict(
            test_data_, saved_model_path=saved_model_path)
        if test_pred is not None:
            assert isinstance(
                test_pred, DataFrame), "test predict result should be a DataFrame"
            add_dataset_type(test_pred, consts.TEST_SET)
            predict_data_output.write(test_pred)
        else:
            logger.warning(
                "test_pred is None, It seems that the runner is not able to predict. Failed to output data")

    elif role.is_arbiter:  # is server
        logger.info("arbiter skip predict")
