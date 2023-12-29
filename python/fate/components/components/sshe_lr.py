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

import logging

from fate.arch import Context
from fate.arch.dataframe import DataFrame
from fate.components.components.utils import consts, tools
from fate.components.core import GUEST, HOST, Role, cpn, params
from fate.ml.glm.hetero.sshe import SSHELogisticRegression

logger = logging.getLogger(__name__)


@cpn.component(roles=[GUEST, HOST], provider="fate")
def sshe_lr(ctx, role):
    ...


@sshe_lr.train()
def train(
    ctx: Context,
    role: Role,
    train_data: cpn.dataframe_input(roles=[GUEST, HOST]),
    validate_data: cpn.dataframe_input(roles=[GUEST, HOST], optional=True),
    epochs: cpn.parameter(type=params.conint(gt=0), default=20, desc="max iteration num"),
    batch_size: cpn.parameter(
        type=params.conint(ge=10),
        default=None,
        desc="batch size, None means full batch, otherwise should be no less than 10, default None",
    ),
    tol: cpn.parameter(type=params.confloat(ge=0), default=1e-4),
    early_stop: cpn.parameter(
        type=params.string_choice(["weight_diff", "diff", "abs"]),
        default="diff",
        desc="early stopping criterion, choose from {weight_diff, diff, abs}, if use weight_diff,"
        "weight will be revealed every epoch",
    ),
    learning_rate: cpn.parameter(type=params.confloat(ge=0), default=0.05, desc="learning rate"),
    reveal_every_epoch: cpn.parameter(
        type=bool, default=False, desc="whether reveal encrypted result every epoch, " "only accept False for now"
    ),
    init_param: cpn.parameter(
        type=params.init_param(),
        default=params.InitParam(method="random_uniform", fit_intercept=True, random_state=None),
        desc="Model param init setting.",
    ),
    threshold: cpn.parameter(
        type=params.confloat(ge=0.0, le=1.0), default=0.5, desc="predict threshold for binary data"
    ),
    reveal_loss_freq: cpn.parameter(
        type=params.conint(ge=1),
        default=1,
        desc="rounds to reveal training loss, " "only effective if `early_stop` is 'loss'",
    ),
    train_output_data: cpn.dataframe_output(roles=[GUEST]),
    output_model: cpn.json_model_output(roles=[GUEST, HOST]),
    warm_start_model: cpn.json_model_input(roles=[GUEST, HOST], optional=True),
):
    logger.info(f"enter sshe lr train")
    init_param = init_param.dict()
    ctx.mpc.init()

    train_model(
        ctx,
        role,
        train_data,
        validate_data,
        train_output_data,
        output_model,
        epochs,
        batch_size,
        learning_rate,
        tol,
        early_stop,
        init_param,
        reveal_every_epoch,
        reveal_loss_freq,
        threshold,
        warm_start_model,
    )


@sshe_lr.predict()
def predict(
    ctx,
    role: Role,
    # threshold: cpn.parameter(type=params.confloat(ge=0.0, le=1.0), default=0.5),
    test_data: cpn.dataframe_input(roles=[GUEST, HOST]),
    input_model: cpn.json_model_input(roles=[GUEST, HOST]),
    test_output_data: cpn.dataframe_output(roles=[GUEST]),
):
    ctx.mpc.init()
    predict_from_model(ctx, role, input_model, test_data, test_output_data)


@sshe_lr.cross_validation()
def cross_validation(
    ctx: Context,
    role: Role,
    cv_data: cpn.dataframe_input(roles=[GUEST, HOST]),
    epochs: cpn.parameter(type=params.conint(gt=0), default=20, desc="max iteration num"),
    batch_size: cpn.parameter(
        type=params.conint(ge=10),
        default=None,
        desc="batch size, None means full batch, otherwise should be no less than 10, default None",
    ),
    tol: cpn.parameter(type=params.confloat(ge=0), default=1e-4),
    early_stop: cpn.parameter(
        type=params.string_choice(["weight_diff", "diff", "abs"]),
        default="diff",
        desc="early stopping criterion, choose from {weight_diff, diff, abs}, if use weight_diff,"
        "weight will be revealed every epoch",
    ),
    learning_rate: cpn.parameter(type=params.confloat(ge=0), default=0.05, desc="learning rate"),
    init_param: cpn.parameter(
        type=params.init_param(),
        default=params.InitParam(method="random_uniform", fit_intercept=True, random_state=None),
        desc="Model param init setting.",
    ),
    threshold: cpn.parameter(
        type=params.confloat(ge=0.0, le=1.0), default=0.5, desc="predict threshold for binary data"
    ),
    reveal_every_epoch: cpn.parameter(
        type=bool, default=False, desc="whether reveal encrypted result every epoch, " "only accept False for now"
    ),
    reveal_loss_freq: cpn.parameter(
        type=params.conint(ge=1),
        default=1,
        desc="rounds to reveal training loss, " "only effective if `early_stop` is 'loss'",
    ),
    cv_param: cpn.parameter(
        type=params.cv_param(),
        default=params.CVParam(n_splits=5, shuffle=False, random_state=None),
        desc="cross validation param",
    ),
    metrics: cpn.parameter(type=params.metrics_param(), default=["auc"]),
    output_cv_data: cpn.parameter(type=bool, default=True, desc="whether output prediction result per cv fold"),
    cv_output_datas: cpn.dataframe_outputs(roles=[GUEST, HOST], optional=True),
):
    init_param = init_param.dict()
    ctx.mpc.init()

    from fate.arch.dataframe import KFold

    kf = KFold(
        ctx, role=role, n_splits=cv_param.n_splits, shuffle=cv_param.shuffle, random_state=cv_param.random_state
    )
    i = 0
    for fold_ctx, (train_data, validate_data) in ctx.on_cross_validations.ctxs_zip(kf.split(cv_data.read())):
        logger.info(f"enter fold {i}")
        module = SSHELogisticRegression(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            tol=tol,
            early_stop=early_stop,
            init_param=init_param,
            threshold=threshold,
            reveal_every_epoch=reveal_every_epoch,
            reveal_loss_freq=reveal_loss_freq,
        )
        module.fit(fold_ctx, train_data, validate_data)
        if output_cv_data:
            if role.is_guest:
                sub_ctx = fold_ctx.sub_ctx("predict_train")
                predict_df = module.predict(sub_ctx, train_data)
                train_predict_result = tools.add_dataset_type(predict_df, consts.TRAIN_SET)
                sub_ctx = fold_ctx.sub_ctx("predict_validate")
                predict_df = module.predict(sub_ctx, validate_data)
                validate_predict_result = tools.add_dataset_type(predict_df, consts.VALIDATE_SET)
                predict_result = DataFrame.vstack([train_predict_result, validate_predict_result])
                next(cv_output_datas).write(df=predict_result)
            elif role.is_host:
                sub_ctx = fold_ctx.sub_ctx("predict_train")
                module.predict(sub_ctx, train_data)
                sub_ctx = fold_ctx.sub_ctx("predict_validate")
                module.predict(sub_ctx, validate_data)
        i += 1


def train_model(
    ctx,
    role,
    train_data,
    validate_data,
    train_output_data,
    output_model,
    epochs,
    batch_size,
    learning_rate,
    tol,
    early_stop,
    init_param,
    reveal_every_epoch,
    reveal_loss_freq,
    threshold,
    input_model,
):
    if input_model is not None:
        logger.info(f"warm start model provided")
        model = input_model.read()
        module = SSHELogisticRegression.from_model(model)
        module.set_epochs(epochs)
        module.set_batch_size(batch_size)

    else:
        module = SSHELogisticRegression(
            epochs=epochs,
            batch_size=batch_size,
            tol=tol,
            early_stop=early_stop,
            learning_rate=learning_rate,
            init_param=init_param,
            threshold=threshold,
            reveal_every_epoch=reveal_every_epoch,
            reveal_loss_freq=reveal_loss_freq,
        )
    # optimizer = optimizer_factory(optimizer_param)
    logger.info(f"sshe lr guest start train")
    sub_ctx = ctx.sub_ctx("train")
    train_data = train_data.read()

    if validate_data is not None:
        logger.info(f"validate data provided")
        validate_data = validate_data.read()

    module.fit(sub_ctx, train_data, validate_data)
    model = module.get_model()
    output_model.write(model, metadata={})

    sub_ctx = ctx.sub_ctx("predict")

    predict_df = module.predict(sub_ctx, train_data)

    if role.is_guest:
        predict_result = tools.add_dataset_type(predict_df, consts.TRAIN_SET)
    if validate_data is not None:
        sub_ctx = ctx.sub_ctx("validate_predict")
        predict_df = module.predict(sub_ctx, validate_data)
        if role.is_guest:
            validate_predict_result = tools.add_dataset_type(predict_df, consts.VALIDATE_SET)
            predict_result = DataFrame.vstack([predict_result, validate_predict_result])
    if role.is_guest:
        train_output_data.write(predict_result)


def predict_from_model(ctx, role, input_model, test_data, test_output_data):
    logger.info(f"sshe lr guest start predict")
    sub_ctx = ctx.sub_ctx("predict")
    model = input_model.read()
    module = SSHELogisticRegression.from_model(model)
    # if module.threshold != 0.5:
    #    module.threshold = threshold
    test_data = test_data.read()
    predict_df = module.predict(sub_ctx, test_data)
    if role.is_guest:
        predict_result = tools.add_dataset_type(predict_df, consts.TEST_SET)
        test_output_data.write(predict_result)
