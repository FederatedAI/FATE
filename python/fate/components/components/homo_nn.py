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
from fate.interface import Context
from fate.components.components.nn.setup.fate_setup import FateSetup
from fate.components.components.nn.nn_setup import NNSetup
from fate.components.components.nn.loader import Loader
from fate.arch.dataframe._dataframe import DataFrame
import logging


logger = logging.getLogger(__name__)


def is_path(s):
    return os.path.exists(s)


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
   
    print('setup conf is {}'.format(setup_conf))
    print('source is {}'.format(source))
    if setup_module != 'fate_setup':
        if source == None:
            # load from default folder
            setup = Loader('fate.components.components.nn.setup.' + setup_module, setup_class, **setup_conf).call_item()
        else:
            setup = Loader(setup_module, setup_class, source=source, **setup_conf).call_item()
        assert isinstance(setup, NNSetup), 'loaded class must be a subclass of NNSetup class, but got {}'.format(type(setup))
    else:
        print('using default fate setup')
        setup = FateSetup(**setup_conf)

    setup.set_context(ctx)
    setup.set_role(role)

    print('setup class is {}'.format(setup))

    if role.is_guest or role.is_host:
        with ctx.sub_ctx("train") as sub_ctx:
            train_data: DataFrame = sub_ctx.reader(train_data).read_dataframe().data
            train_data: pd.DataFrame = train_data.as_pd_df()
            print('train data is {}'.format(train_data))
            setup.set_cpn_input_data(train_data)
            # get trainer
            trainer = setup.setup()
            trainer.train()

    elif role.is_arbiter:
        # get trainer
        server_trainer = setup.setup()
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
    pass
