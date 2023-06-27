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
)
import os
import pandas as pd
from fate.interface import Context
from fate.components.components.nn.runner.default_runner import DefaultRunner
from fate.components.components.nn.nn_runner import NNRunner, NNInput, NNOutput
from fate.components.components.nn.loader import Loader
from fate.arch.dataframe import PandasReader
from fate.components.components.utils import consts
from fate.components.components.utils.predict_format import LABEL
import logging


logger = logging.getLogger(__name__)


FATE_TEST_PATH = '/home/cwj/FATE/playground/test_output_path'


def is_path(s):
    return os.path.exists(s)

"""
Input Functions
"""


def prepare_runner_class(runner_module, runner_class, runner_conf, source):
    logger.info('runner conf is {}'.format(runner_conf))
    logger.info('source is {}'.format(source))
    if runner_module != 'fate_runner':
        if source == None:
            # load from default folder
            runner = Loader('fate.components.components.nn.runner.' + runner_module, runner_class, **runner_conf)()
        else:
            runner = Loader(runner_module, runner_class, source=source, **runner_conf)()
        assert isinstance(runner, NNRunner), 'loaded class must be a subclass of NNRunner class, but got {}'.format(type(runner))
    else:
        logger.info('using default fate runner')
        runner = DefaultRunner(**runner_conf)

    return runner


def prepare_context_and_role(runner, ctx, role, sub_ctx_name):
    with ctx.sub_ctx(sub_ctx_name) as sub_ctx:
        # set context
        runner.set_context(sub_ctx)
        runner.set_role(role)
        return sub_ctx


def get_input_data(sub_ctx, stage, cpn_input_data, input_type='df'):
    if stage == 'train':
        train_data, validate_data = cpn_input_data
        if input_type == 'df':
            train_data = sub_ctx.reader(train_data).read_dataframe().data
            if validate_data is not None:
                validate_data = sub_ctx.reader(validate_data).read_dataframe().data

        return NNInput(train_data=train_data, validate_data=validate_data)
    
    elif stage == 'predict':
        test_data = cpn_input_data
        test_data = sub_ctx.reader(test_data).read_dataframe().data
        return NNInput(test_data=test_data)
    else:
        raise ValueError(f'Unknown stage {stage}')


""""
Output functions
"""

def model_output(runner_module,
                 runner_class,
                 runner_conf,
                 source,
                 model_output_path
                ):
    return {
        'runner_module': runner_module,
        'runner_class': runner_class,
        'runner_conf': runner_conf,
        'source': source,
        'model_output_path': model_output_path
    }


def write_output_df(ctx, result_df: pd.DataFrame, output_data_cls, match_id_name, sample_id_name):
    
    reader = PandasReader(sample_id_name=sample_id_name, match_id_name=match_id_name, label_name=LABEL, dtype="object")
    data = reader.to_frame(ctx, result_df)
    ctx.writer(output_data_cls).write_dataframe(data)


def handle_nn_output(ctx, nn_output: NNOutput, output_class, stage):

    if nn_output is None:
        logger.warning('runner output is None in stage:{}, skip processing'.format(stage))

    elif isinstance(nn_output, NNOutput):
        if stage == consts.TRAIN:
            
            if nn_output.train_result is None and nn_output.validate_result is None:
                raise ValueError('train result and validate result are both None in the NNOutput: {}'.format(nn_output))
            
            df_train, df_val = None, None
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
                raise ValueError('test result not found in the NNOutput: {}'.format(nn_output))
            write_output_df(ctx, nn_output.test_result, output_class, nn_output.match_id_name, nn_output.sample_id_name)
    else:
        logger.warning('train output is not NNOutput, but {}'.format(type(nn_output)))


@cpn.component(roles=[GUEST, HOST, ARBITER])
def homo_nn(ctx, role):
    ...


@homo_nn.train()
@cpn.artifact("train_data", type=Input[DatasetArtifact], roles=[GUEST, HOST], desc="training data")
@cpn.artifact("validate_data", type=Input[DatasetArtifact], optional=True, roles=[GUEST, HOST], desc="validation data")
@cpn.parameter("runner_module", type=str, default='default_runner', desc="name of your runner script")
@cpn.parameter("runner_class", type=str, default='DefaultRunner', desc="class name of your runner class")
@cpn.parameter("source", type=str, default=None, desc="path to your runner script folder")
@cpn.parameter("runner_conf", type=dict, default={}, desc="the parameter dict of the NN runner class")
@cpn.artifact("train_output_data", type=Output[DatasetArtifact], roles=[GUEST, HOST, ARBITER])
@cpn.artifact("train_output_metric", type=Output[LossMetrics], roles=[GUEST, HOST, ARBITER])
@cpn.artifact("output_model", type=Output[ModelArtifact], roles=[GUEST, HOST])
def train(
    ctx: Context,
    role: Role,
    train_data,
    validate_data,
    runner_module,
    runner_class,
    runner_conf,
    source,
    train_output_data,
    train_output_metric,
    output_model,
):
   
    runner: NNRunner = prepare_runner_class(runner_module, runner_class, runner_conf, source)
    sub_ctx = prepare_context_and_role(runner, ctx, role, consts.TRAIN)

    if role.is_guest or role.is_host:  # is client

        input_data = get_input_data(sub_ctx, consts.TRAIN, [train_data, validate_data])
        input_data.fate_save_path = FATE_TEST_PATH
        ret: NNOutput = runner.train(input_data=input_data)
        handle_nn_output(sub_ctx, ret, train_output_data, consts.TRAIN)

        output_conf = model_output(runner_module,
                                   runner_class,
                                   runner_conf,
                                   source,
                                   FATE_TEST_PATH)
        import json
        path = '/home/cwj/FATE/playground/test_output_model/'
        json.dump(output_conf, open(path + str(role.name) + '_conf.json', 'w'), indent=4)

        with output_model as model_writer:
            model_writer.write_model("homo_nn", {}, metadata={})
        
    elif role.is_arbiter:  # is server
        runner.train()


@homo_nn.predict()
@cpn.artifact("input_model", type=Input[ModelArtifact], roles=[GUEST, HOST])
@cpn.artifact("test_data", type=Input[DatasetArtifact], optional=False, roles=[GUEST, HOST])
@cpn.artifact("test_output_data", type=Output[DatasetArtifact], roles=[GUEST, HOST])
def predict(
    ctx,
    role: Role,
    test_data,
    input_model,
    test_output_data,
):

    if role.is_guest or role.is_host:  # is client

        import json
        path = '/home/cwj/FATE/playground/test_output_model/'
        model_conf = json.load(open(path + str(role.name) + '_conf.json', 'r'))
        runner_module = model_conf['runner_module']
        runner_class = model_conf['runner_class']
        runner_conf = model_conf['runner_conf']
        source = model_conf['source']

        runner: NNRunner = prepare_runner_class(runner_module, runner_class, runner_conf, source)
        sub_ctx = prepare_context_and_role(runner, ctx, role, consts.PREDICT)
        input_data = get_input_data(sub_ctx, consts.PREDICT, test_data)
        pred_rs = runner.predict(input_data)
        handle_nn_output(sub_ctx, pred_rs, test_output_data, consts.PREDICT)

    elif role.is_arbiter:  # is server
        logger.info('arbiter skip predict')
