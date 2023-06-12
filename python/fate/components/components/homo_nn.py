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
import os
import pandas as pd
from typing import Literal
from fate.interface import Context
from fate.components.components.nn.setup.fate_setup import FateSetup
from fate.components.components.nn.nn_setup import NNRunner, SetupReturn, ComponentInputData
from fate.components.components.nn.loader import Loader
from fate.arch.dataframe._dataframe import DataFrame
import logging


logger = logging.getLogger(__name__)


def is_path(s):
    return os.path.exists(s)


def prepare_setup_class(setup_module, setup_class, setup_conf, source):
    print('setup conf is {}'.format(setup_conf))
    print('source is {}'.format(source))
    if setup_module != 'fate_setup':
        if source == None:
            # load from default folder
            setup = Loader('fate.components.components.nn.setup.' + setup_module, setup_class, **setup_conf)()
        else:
            setup = Loader(setup_module, setup_class, source=source, **setup_conf)()
        assert isinstance(setup, NNRunner), 'loaded class must be a subclass of NNSetup class, but got {}'.format(type(setup))
    else:
        print('using default fate setup')
        setup = FateSetup(**setup_conf)

    return setup


def transform_input_dataframe(sub_ctx, data):

    df_: DataFrame = sub_ctx.reader(data).read_dataframe().data
    df: pd.DataFrame = df_.as_pd_df()
    return df


def prepare_context_and_role(setup, ctx, role, sub_ctx_name):
    with ctx.sub_ctx(sub_ctx_name) as sub_ctx:
        # set context
        setup.set_context(sub_ctx)
        setup.set_role(role)
        return sub_ctx


def process_setup(setup, stage=Literal['train', 'predict']):
    setup_ret = setup.setup(stage)
    print('setup class is {}'.format(setup))
    if not isinstance(setup_ret, SetupReturn):
        raise ValueError(f'The return of your setup class must be a SetupReturn Instance, but got {setup_ret}')
    return setup_ret


def handle_client(setup, sub_ctx, stage, cpn_input_data):
    if stage == 'train':
        train_data, validate_data = cpn_input_data
        train_data = transform_input_dataframe(sub_ctx, train_data)
        if validate_data is not None:
            validate_data = transform_input_dataframe(sub_ctx, validate_data)
        setup.set_cpn_input_data(ComponentInputData(train_data, validate_data))
    elif stage == 'predict':
        test_data = cpn_input_data
        test_data = transform_input_dataframe(sub_ctx, test_data)
        setup.set_cpn_input_data(ComponentInputData(test_data=test_data))
    else:
        raise ValueError(f'Unknown stage {stage}')
    setup_ret = process_setup(setup, stage)
    return setup_ret


def handle_server(setup, stage):
    return process_setup(setup, stage)['trainer']


def update_output_dir(trainer):
    
    FATE_TEST_PATH = '/home/cwj/FATE/playground/test_output_path'
    # default trainer
    trainer.args.output_dir = FATE_TEST_PATH


def model_output(setup_module,
                 setup_class,
                 setup_conf,
                 source,
                 model_output_path
                ):
    return {
        'setup_module': setup_module,
        'setup_class': setup_class,
        'setup_conf': setup_conf,
        'source': source,
        'model_output_path': model_output_path
    }


@cpn.component(roles=[GUEST, HOST, ARBITER])
def homo_nn(ctx, role):
    ...


@homo_nn.train()
@cpn.artifact("train_data", type=Input[DatasetArtifact], roles=[GUEST, HOST], desc="training data")
@cpn.artifact("validate_data", type=Input[DatasetArtifact], optional=True, roles=[GUEST, HOST], desc="validation data")
@cpn.parameter("setup_module", type=str, default='fate_setup', desc="name of your setup script")
@cpn.parameter("setup_class", type=str, default='FateSetup', desc="class name of your setup class")
@cpn.parameter("source", type=str, default=None, desc="path to your setup script folder")
@cpn.parameter("setup_conf", type=dict, default={}, desc="the parameter dict of the NN setup class")
@cpn.artifact("train_output_data", type=Output[DatasetArtifact], roles=[GUEST, HOST])
@cpn.artifact("train_output_metric", type=Output[LossMetrics], roles=[ARBITER])
@cpn.artifact("output_model", type=Output[ModelArtifact], roles=[GUEST, HOST])
def train(
    ctx: Context,
    role: Role,
    train_data,
    validate_data,
    setup_module,
    setup_class,
    setup_conf,
    source,
    train_output_data,
    train_output_metric,
    output_model,
):
   
    setup = prepare_setup_class(setup_module, setup_class, setup_conf, source)
    sub_ctx = prepare_context_and_role(setup, ctx, role, "train")

    if role.is_guest or role.is_host:  # is client
        setup_ret = handle_client(setup, sub_ctx, 'train', [train_data, validate_data])
        client_trainer = setup_ret['trainer']
        update_output_dir(client_trainer)  # update output dir
        client_trainer.train()

        output_conf = model_output(setup_module,
                                   setup_class,
                                   setup_conf,
                                   source,
                                   client_trainer.args.output_dir)

        print('model output is {}'.format(output_conf))
        client_trainer.save_model()
        import json
        path = '/home/cwj/FATE/playground/test_output_model/'
        json.dump(output_conf, open(path + str(role.name) + '_conf.json', 'w'), indent=4)

        with output_model as model_writer:
            model_writer.write_model("homo_nn", {}, metadata={})
        
    elif role.is_arbiter:  # is server
        server_trainer = handle_server(setup, 'train')
        server_trainer.train()


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
        setup_module = model_conf['setup_module']
        setup_class = model_conf['setup_class']
        setup_conf = model_conf['setup_conf']
        source = model_conf['source']
    
        setup = prepare_setup_class(setup_module, setup_class, setup_conf, source)
        sub_ctx = prepare_context_and_role(setup, ctx, role, "predict")
        setup_ret = handle_client(setup, sub_ctx, 'predict', test_data)
        to_predict_dataset = setup_ret['test_set']
        client_trainer = setup_ret['trainer']

        if to_predict_dataset is None:
            raise ValueError('The return of your setup class in the training stage must have "test_set" in the predict stage')
        
        pred_rs = client_trainer.predict(to_predict_dataset)
        print(f'predict result is {pred_rs}')

    elif role.is_arbiter:  # is server
        print('arbiter skip predict')