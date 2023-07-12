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
from fate.arch import Context
from fate.ml.glm.homo_lr.client import HomoLRClient
from fate.ml.glm.homo_lr.server import HomoLRServer
from fate.components.core import ARBITER, GUEST, HOST, Role, cpn, params
from fate.components.components.utils import consts
from fate.ml.utils.model_io import ModelIO


logger = logging.getLogger(__name__)



@cpn.component(roles=[GUEST, HOST, ARBITER])
def homo_lr(ctx, role):
    ...


@homo_lr.train()
def train(
    ctx: Context,
    role: Role,
    train_data: cpn.dataframe_input(roles=[GUEST, HOST]),
    validate_data: cpn.dataframe_input(roles=[GUEST, HOST], optional=True),
    learning_rate_scheduler: cpn.parameter(type=params.lr_scheduler_param(),
                                            default=params.LRSchedulerParam(method="linear",
                                                                            scheduler_params={"start_factor": 1.0}),
                                            desc="learning rate scheduler, "
                                                "select method from {'step', 'linear', 'constant'}"
                                                "for list of configurable arguments, "
                                                "refer to torch.optim.lr_scheduler"),
    epochs: cpn.parameter(type=params.conint(gt=0), default=20,
                            desc="max iteration num"),
    batch_size: cpn.parameter(type=params.conint(ge=-1), default=100,
                                desc="batch size, "
                                    "value less or equals to 0 means full batch"),
    optimizer: cpn.parameter(type=params.optimizer_param(),
                                default=params.OptimizerParam(method="sgd", penalty='l2', alpha=1.0,
                                                            optimizer_params={"lr": 1e-2, "weight_decay": 0})),
    init_param: cpn.parameter(type=params.init_param(),
                                default=params.InitParam(method='zeros', fit_intercept=True),
                                desc="Model param init setting."),
    threshold: cpn.parameter(type=params.confloat(ge=0.0, le=1.0), default=0.5,
                                desc="predict threshold for binary data"),
    ovr: cpn.parameter(type=bool, default=False,
                                desc="predict threshold for binary data"),
    label_num: cpn.parameter(type=params.conint(ge=2), default=None),
    train_output_data: cpn.dataframe_output(roles=[GUEST, HOST]),
    train_input_model: cpn.json_model_input(roles=[GUEST, HOST], optional=True),
    train_output_model: cpn.json_model_output(roles=[GUEST, HOST])
):

    sub_ctx = ctx.sub_ctx(consts.TRAIN)

    if role.is_guest or role.is_host:  # is client
        
        logger.info('homo lr component: client start training')
        logger.info('optim param {} init param {}'.format(optimizer.dict(), init_param.dict()))

        client = HomoLRClient(epochs=epochs, batch_size=batch_size, optimizer_param=optimizer.dict(), init_param=init_param.dict(), 
                              learning_rate_scheduler=0.01, threshold=threshold, ovr=ovr, label_num=label_num)
        train_df = train_data.read()
        validate_df = validate_data.read() if validate_data else None
        client.fit(sub_ctx, train_df, validate_df)
        model_dict = client.get_model().dict()
        
        train_rs = client.predict(sub_ctx, train_df)
        if validate_df:
            validate_rs = client.predict(sub_ctx, validate_df)
            ret_df = train_rs.vstack(validate_rs)
        else:
            ret_df = train_rs

        train_output_data.write(ret_df)
        train_output_model.write(model_dict, metadata=model_dict['meta'])

    elif role.is_arbiter:  # is server
        logger.info('homo lr component: server start training')
        server = HomoLRServer()
        server.fit(sub_ctx)


@homo_lr.predict()
def predict(
    ctx,
    role: Role,
    test_data: cpn.dataframe_input(roles=[GUEST, HOST]),
    batch_size: cpn.parameter(type=params.conint(ge=-1), default=100,
                                desc="batch size, "
                                "value less or equals to 0 means full batch"),
    threshold: cpn.parameter(type=params.confloat(ge=0.0, le=1.0), default=0.5,
                                desc="predict threshold for binary data"),
    predict_input_model: cpn.json_model_input(roles=[GUEST, HOST]),
    test_output_data: cpn.dataframe_output(roles=[GUEST, HOST])
):

    if role.is_guest or role.is_host:  # is client

        client = HomoLRClient(batch_size=batch_size, threshold=threshold)
        model_input = predict_input_model.read()
        model_data = ModelIO.from_dict(model_input)
        logger.info('model input is {}'.format(model_input))
        pred_rs = client.predict(ctx, test_data.read())
        

    elif role.is_arbiter:  # is server
        logger.info("arbiter skip predict")
