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
from fate.components.components.nn.loader import Loader
from fate.components.components.nn.nn_runner import NNRunner
from fate.components.components.nn.runner.default_runner import DefaultRunner
from fate.components.components.utils import consts
from fate.components.core import ARBITER, GUEST, HOST, Role, cpn
from fate.arch.dataframe import DataFrame
from fate.components.components.utils.tools import add_dataset_type

logger = logging.getLogger(__name__)


def is_path(s):
    return os.path.exists(s)


"""
Input Functions
"""


def prepare_runner_class(runner_module, runner_class, runner_conf, source):
    logger.info("runner conf is {}".format(runner_conf))
    logger.info("source is {}".format(source))
    if runner_module != "fate_runner":
        if source is None:
            # load from default folder
            runner = Loader(
                "fate.components.components.nn.runner." +
                runner_module,
                runner_class,
                **runner_conf)()
        else:
            runner = Loader(
                runner_module,
                runner_class,
                source=source,
                **runner_conf)()
        assert isinstance(
            runner, NNRunner), "loaded class must be a subclass of NNRunner class, but got {}".format(
            type(runner))
    else:
        logger.info("using default fate runner")
        runner = DefaultRunner(**runner_conf)

    return runner


def prepare_context_and_role(runner, ctx, role, sub_ctx_name):
    sub_ctx = ctx.sub_ctx(sub_ctx_name)
    runner.set_context(sub_ctx)
    runner.set_role(role)


def get_input_data(stage, cpn_input_data):

    if stage == 'train':
        train_data, validate_data = cpn_input_data
        train_data = train_data.read()
        if validate_data is not None:
            validate_data = validate_data.read()

        return train_data, validate_data

    elif stage == 'predict':
        test_data = cpn_input_data
        test_data = test_data.read()
        return test_data
    else:
        raise ValueError(f"Unknown stage {stage}")


""""
Output functions
"""


def get_model_output_conf(runner_module,
                          runner_class,
                          runner_conf,
                          source,
                          ):
    return {
        "runner_module": runner_module,
        "runner_class": runner_class,
        "runner_conf": runner_conf,
        "source": source,
    }


def prepared_saved_conf(
        model_conf,
        runner_class,
        runner_module,
        runner_conf,
        source):

    logger.info("loaded model_conf is: {}".format(model_conf))
    if "source" in model_conf:
        if source is None:
            source = model_conf["source"]

    runner_class_, runner_module_ = model_conf['runner_class'], model_conf['runner_module']
    if runner_class_ == runner_class and runner_module_ == runner_module:
        if "runner_conf" in model_conf:
            saved_conf = model_conf['runner_conf']
            saved_conf.update(runner_conf)
            runner_conf = saved_conf
            logger.info("runner_conf is updated: {}".format(runner_conf))
    else:
        logger.warning(
            "runner_class or runner_module is not equal to the saved model, "
            "use the new runner_conf, runner_class and runner module to train the model,\
                        saved module & class: {} {}, new module & class: {} {}".format(
                runner_module_, runner_class_, runner_module, runner_class))

    return runner_conf, source, runner_class, runner_module


@cpn.component(roles=[GUEST, HOST, ARBITER])
def homo_nn(ctx, role):
    ...


@homo_nn.train()
def train(
    ctx: Context,
    role: Role,
    train_data: cpn.dataframe_input(roles=[GUEST, HOST]),
    validate_data: cpn.dataframe_input(roles=[GUEST, HOST], optional=True),
    runner_module: cpn.parameter(type=str, default="default_runner", desc="name of your runner script"),
    runner_class: cpn.parameter(type=str, default="DefaultRunner", desc="class name of your runner class"),
    runner_conf: cpn.parameter(type=dict, default={}, desc="the parameter dict of the NN runner class"),
    source: cpn.parameter(type=str, default=None, desc="path to your runner script folder"),
    train_data_output: cpn.dataframe_output(roles=[GUEST, HOST], optional=True),
    train_model_output: cpn.model_directory_output(roles=[GUEST, HOST], optional=True),
    train_model_input: cpn.model_directory_input(roles=[GUEST, HOST], optional=True)
):


    if role.is_guest or role.is_host:  # is client
        
        if train_model_input is not None:
            model_conf = train_model_input.get_metadata()
            runner_conf, source, runner_class, runner_module = prepared_saved_conf(
                model_conf, runner_class, runner_module, runner_conf, source)
            saved_model_path = str(train_model_input.get_directory())
        else:
            saved_model_path = None

        runner: NNRunner = prepare_runner_class(
            runner_module, runner_class, runner_conf, source)
        prepare_context_and_role(runner, ctx, role, consts.TRAIN)

        output_dir = str(train_model_output.get_directory())
        train_data_, validate_data_ = get_input_data(
            consts.TRAIN, [train_data, validate_data])
        runner.train(train_data_, validate_data_, output_dir, saved_model_path)

        logger.info('Predicting Train & Validate Data')
        train_pred = runner.predict(train_data_, saved_model_path)
        if train_pred is not None:
            assert isinstance(
                train_pred, DataFrame), "train predict result should be a DataFrame"
            add_dataset_type(train_pred, consts.TRAIN_SET)

            if validate_data_ is not None:
                validate_pred = runner.predict(validate_data_)
                assert isinstance(
                    validate_pred, DataFrame), "validate predict result should be a DataFrame"
                add_dataset_type(validate_pred, consts.VALIDATE_SET)
                output_df = DataFrame.vstack([train_pred, validate_pred])
            else:
                output_df = train_pred
            logger.info('write result dataframe')
            train_data_output.write(output_df)
        else:
            logger.warning(
                "train_pred is None, It seems that the runner is not able to predict. Failed to output data")

        output_conf = get_model_output_conf(runner_module,
                                            runner_class,
                                            runner_conf,
                                            source
                                            )
        train_model_output.write_metadata(output_conf)

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
