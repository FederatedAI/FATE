#
#  Copyright 2023 The FATE Authors. All Rights Reserved.
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

import json
import logging

from fate.arch import Context
from fate.arch.dataframe import DataFrame
from fate.components.core import ARBITER, GUEST, HOST, Role, cpn, params

logger = logging.getLogger(__name__)


@cpn.component(roles=[GUEST, HOST, ARBITER])
def coordinated_linr(ctx, role):
    ...


@coordinated_linr.train()
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
        tol: cpn.parameter(type=params.confloat(ge=0), default=1e-4),
        early_stop: cpn.parameter(type=params.string_choice(["weight_diff", "diff", "abs"]), default="diff",
                                  desc="early stopping criterion, choose from {weight_diff, diff, abs, val_metrics}"),
        init_param: cpn.parameter(type=params.init_param(),
                                  default=params.InitParam(method='zeros', fit_intercept=True),
                                  desc="Model param init setting."),
        train_output_data: cpn.dataframe_output(roles=[GUEST, HOST]),
        output_model: cpn.json_model_output(roles=[GUEST, HOST, ARBITER]),
):
    logger.info(f"enter coordinated linr train")
    # temp code start
    optimizer = optimizer.dict()
    learning_rate_scheduler = learning_rate_scheduler.dict()
    init_param = init_param.dict()
    # temp code end
    if role.is_guest:
        train_guest(
            ctx, train_data, validate_data, train_output_data, output_model, epochs,
            batch_size, optimizer, learning_rate_scheduler, init_param
        )
    elif role.is_host:
        train_host(
            ctx, train_data, validate_data, train_output_data, output_model, epochs,
            batch_size, optimizer, learning_rate_scheduler, init_param
        )
    elif role.is_arbiter:
        train_arbiter(ctx, epochs, early_stop, tol, batch_size, optimizer, learning_rate_scheduler, output_model)


@coordinated_linr.predict()
def predict(
        ctx,
        role: Role,
        test_data: cpn.dataframe_input(roles=[GUEST, HOST]),
        input_model: cpn.json_model_input(roles=[GUEST, HOST]),
        test_output_data: cpn.dataframe_output(roles=[GUEST, HOST])
):
    if role.is_guest:
        predict_guest(ctx, input_model, test_data, test_output_data)
    if role.is_host:
        predict_host(ctx, input_model, test_data, test_output_data)


def train_guest(ctx, train_data, validate_data, train_output_data, output_model, epochs,
                batch_size, optimizer_param, learning_rate_param, init_param):
    logger.info(f"coordinated linr guest start train")
    from fate.ml.glm.coordinated_linr import CoordinatedLinRModuleGuest
    # optimizer = optimizer_factory(optimizer_param)
    sub_ctx = ctx.sub_ctx("train")
    module = CoordinatedLinRModuleGuest(epochs=epochs, batch_size=batch_size,
                                        optimizer_param=optimizer_param, learning_rate_param=learning_rate_param,
                                        init_param=init_param)
    train_data = train_data.read()
    if validate_data is not None:
        validate_data = validate_data.read()
    module.fit(sub_ctx, train_data, validate_data)
    model = module.get_model()
    output_model.write(model, metadata={"optimizer_param": optimizer_param,
                                        "learning_rate_param": learning_rate_param})

    sub_ctx = ctx.sub_ctx("predict")

    predict_score = module.predict(sub_ctx, train_data)
    predict_result = transform_to_predict_result(train_data, predict_score,
                                                 data_type="train")
    if validate_data is not None:
        predict_score = module.predict(sub_ctx, validate_data)
        validate_predict_result = transform_to_predict_result(validate_data, predict_score,
                                                              data_type="validate")
        predict_result = DataFrame.vstack([predict_result, validate_predict_result])
    train_output_data.write(predict_result)


def train_host(ctx, train_data, validate_data, train_output_data, output_model, epochs, batch_size,
               optimizer_param, learning_rate_param, init_param):
    logger.info(f"coordinated linr host start train")

    from fate.ml.glm.coordinated_linr import CoordinatedLinRModuleHost
    # optimizer = optimizer_factory(optimizer_param)

    sub_ctx = ctx.sub_ctx("train")
    module = CoordinatedLinRModuleHost(epochs=epochs, batch_size=batch_size,
                                       optimizer_param=optimizer_param, learning_rate_param=learning_rate_param,
                                       init_param=init_param)
    train_data = train_data.read()
    if validate_data is not None:
        validate_data = validate_data.read()
    module.fit(sub_ctx, train_data, validate_data)
    model = module.get_model()
    output_model.write(model, metadata={"optimizer_param": optimizer_param,
                                        "learning_rate_param": learning_rate_param})

    sub_ctx = ctx.sub_ctx("predict")
    module.predict(sub_ctx, train_data)
    if validate_data is not None:
        module.predict(sub_ctx, validate_data)


def train_arbiter(ctx, epochs, early_stop, tol, batch_size, optimizer_param,
                  learning_rate_param, output_model):
    logger.info(f"coordinated linr arbiter start train")

    from fate.ml.glm.coordinated_linr import CoordinatedLinRModuleArbiter

    sub_ctx = ctx.sub_ctx("train")
    module = CoordinatedLinRModuleArbiter(epochs=epochs, early_stop=early_stop, tol=tol, batch_size=batch_size,
                                          optimizer_param=optimizer_param, learning_rate_param=learning_rate_param,
                                          )
    module.fit(sub_ctx)

    model = module.get_model()
    output_model.write(model, metadata={"optimizer_param": optimizer_param,
                                        "learning_rate_param": learning_rate_param})


def predict_guest(ctx, input_model, test_data, test_output_data):
    from fate.ml.glm.coordinated_linr import CoordinatedLinRModuleGuest

    sub_ctx = ctx.sub_ctx("predict")
    model = input_model.read()

    module = CoordinatedLinRModuleGuest.from_model(model)
    test_data = test_data.read()
    predict_score = module.predict(sub_ctx, test_data)
    predict_result = transform_to_predict_result(test_data, predict_score, data_type="predict")
    test_output_data.write(predict_result)


def predict_host(ctx, input_model, test_data, test_output_data):
    from fate.ml.glm.coordinated_linr import CoordinatedLinRModuleHost

    sub_ctx = ctx.sub_ctx("predict")
    model = input_model.read()
    module = CoordinatedLinRModuleHost.from_model(model)
    test_data = test_data.read()
    module.predict(sub_ctx, test_data)


def transform_to_predict_result(test_data, predict_score, data_type="test"):
    df = test_data.create_frame(with_label=True, with_weight=False)
    pred_res = test_data.create_frame(with_label=False, with_weight=False)
    pred_res["predict_result"] = predict_score
    df[["predict_result", "predict_score", "predict_detail", "type"]] = pred_res.apply_row(lambda v: [
        v[0],
        v[0],
        json.dumps({"label": v[0]}),
        data_type],
                                                                                           enable_type_align_checking=False)
    return df
