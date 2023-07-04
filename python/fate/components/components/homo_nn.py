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

import pandas as pd
from fate.arch import Context
from fate.arch.dataframe import PandasReader
from fate.components.components.nn.loader import Loader
from fate.components.components.nn.nn_runner import NNInput, NNOutput, NNRunner
from fate.components.components.nn.runner.default_runner import DefaultRunner
from fate.components.components.utils import consts
from fate.components.components.utils.predict_format import LABEL
from fate.components.core import ARBITER, GUEST, HOST, Role, cpn

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
        if source == None:
            # load from default folder
            runner = Loader("fate.components.components.nn.runner." + runner_module, runner_class, **runner_conf)()
        else:
            runner = Loader(runner_module, runner_class, source=source, **runner_conf)()
        assert isinstance(runner, NNRunner), "loaded class must be a subclass of NNRunner class, but got {}".format(
            type(runner)
        )
    else:
        logger.info("using default fate runner")
        runner = DefaultRunner(**runner_conf)

    return runner


def prepare_context_and_role(runner, ctx, role, sub_ctx_name):
    sub_ctx = ctx.sub_ctx(sub_ctx_name)
    runner.set_context(sub_ctx)
    runner.set_role(role)
    return sub_ctx


def get_input_data(stage, cpn_input_data, fate_save_path='./', saved_model_path=None,
                  input_type='df',):
    if stage == 'train':
        train_data, validate_data = cpn_input_data
        if input_type == "df":
            train_data = train_data.read()
            if validate_data is not None:
                validate_data = validate_data.read()

        return NNInput(train_data=train_data, validate_data=validate_data, 
                       fate_save_path=fate_save_path, saved_model_path=saved_model_path)
    
    elif stage == 'predict':
        test_data = cpn_input_data
        test_data = test_data.read()
        return NNInput(test_data=test_data,  
                       fate_save_path=fate_save_path, saved_model_path=saved_model_path)
    else:
        raise ValueError(f"Unknown stage {stage}")


""""
Output functions
"""

def get_model_output_conf(runner_module,
                 runner_class,
                 runner_conf,
                 source,
                 model_output_path
                ):
    return {
        "runner_module": runner_module,
        "runner_class": runner_class,
        "runner_conf": runner_conf,
        "source": source,
        "saved_model_path": model_output_path,
    }


def write_output_df(ctx, result_df: pd.DataFrame, output_data_cls, match_id_name, sample_id_name):

    reader = PandasReader(sample_id_name=sample_id_name, match_id_name=match_id_name, label_name=LABEL, dtype="object")
    data = reader.to_frame(ctx, result_df)
    output_data_cls.write(data)


def handle_nn_output(ctx, nn_output: NNOutput, output_class, stage):

    if nn_output is None:
        logger.warning("runner output is None in stage:{}, skip processing".format(stage))

    elif isinstance(nn_output, NNOutput):
        if stage == consts.TRAIN:

            if nn_output.train_result is None and nn_output.validate_result is None:
                raise ValueError(
                    "train result and validate result are both None in the NNOutput: {}".format(nn_output)
                )

            df_train, df_val = nn_output.train_result, nn_output.validate_result

            match_id_name, sample_id_name = nn_output.match_id_name, nn_output.sample_id_name
            if df_train is not None and df_val is not None:
                df_train_val = pd.concat([df_train, df_val], axis=0)
                df_train_val.match_id_name = df_train.match_id_name
                write_output_df(ctx, df_train_val, output_class, match_id_name, sample_id_name)
            elif df_train is not None:
                write_output_df(ctx, df_train, output_class, match_id_name, sample_id_name)
            elif df_val is not None:
                write_output_df(ctx, df_val, output_class, match_id_name, sample_id_name)
        if stage == consts.PREDICT:
            if nn_output.test_result is None:
                raise ValueError("test result not found in the NNOutput: {}".format(nn_output))
            write_output_df(
                ctx, nn_output.test_result, output_class, nn_output.match_id_name, nn_output.sample_id_name
            )
    else:
        logger.warning("train output is not NNOutput, but {}, fail to output dataframe".format(type(nn_output)))


def prepared_saved_conf(model_conf, runner_class, runner_module, runner_conf, source):

    logger.info("loaded model_conf is: {}".format(model_conf))
    if "saved_model_path" in model_conf:
        saved_model_path = model_conf["saved_model_path"]
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
        logger.warning("runner_class or runner_module is not equal to the saved model, "
                        "use the new runner_conf, runner_class and runner module to train the model,\
                        saved module & class: {} {}, new module & class: {} {}".format(runner_module_, runner_class_, runner_module, runner_class))

    return runner_conf, source, runner_class, runner_module, saved_model_path


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
    train_data_output: cpn.dataframe_output(roles=[GUEST, HOST]),
    train_model_output: cpn.model_directory_output(roles=[GUEST, HOST]),
    train_model_input: cpn.model_directory_input(roles=[GUEST, HOST], optional=True),
):

    runner: NNRunner = prepare_runner_class(runner_module, runner_class, runner_conf, source)
    sub_ctx = prepare_context_and_role(runner, ctx, role, consts.TRAIN)

    if role.is_guest or role.is_host:  # is client
        
        saved_model_path=None
        if train_model_input is not None:
            model_conf = train_model_input.get_metadata()
            runner_conf, source, runner_class, runner_module, saved_model_path = prepared_saved_conf(model_conf, runner_class, runner_module, runner_conf, source)

        output_path = train_model_output.get_directory()
        input_data = get_input_data(consts.TRAIN, [train_data, validate_data], output_path, saved_model_path)
        ret: NNOutput = runner.train(input_data=input_data)
        logger.info("train result: {}".format(ret))
        handle_nn_output(sub_ctx, ret, train_data_output, consts.TRAIN)
        output_conf = get_model_output_conf(runner_module,
                                            runner_class,
                                            runner_conf,
                                            source,
                                            output_path)
        logger.info("output_path: {}".format(output_conf))
        train_model_output.write_metadata(output_conf)
        
    elif role.is_arbiter:  # is server
        runner.train()


@homo_nn.predict()
def predict(
    ctx,
    role: Role,
    test_data: cpn.dataframe_input(roles=[GUEST, HOST]),
    predict_model_input: cpn.model_directory_input(roles=[GUEST, HOST]),
    predict_data_output: cpn.dataframe_output(roles=[GUEST, HOST])
):

    if role.is_guest or role.is_host:  # is client

        model_conf = predict_model_input.get_metadata()
        runner_module = model_conf['runner_module']
        runner_class = model_conf['runner_class']
        runner_conf = model_conf['runner_conf']
        source = model_conf['source']
        saved_model_path = model_conf["saved_model_path"]

        runner: NNRunner = prepare_runner_class(runner_module, runner_class, runner_conf, source)
        sub_ctx = prepare_context_and_role(runner, ctx, role, consts.PREDICT)
        input_data = get_input_data(consts.PREDICT, test_data, saved_model_path=saved_model_path)
        ret: NNOutput = runner.predict(input_data)
        handle_nn_output(sub_ctx, ret, predict_data_output, consts.PREDICT)

    elif role.is_arbiter:  # is server
        logger.info("arbiter skip predict")
